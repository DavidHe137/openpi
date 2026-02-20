import argparse
import os
import statistics
import threading
import time

import jax
import numpy as np
import websockets.sync.client
from dataclasses import asdict
from openpi_client import msgpack_numpy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferType

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "


def _print_stats(label: str, samples_ms: list[float]) -> None:
    if not samples_ms:
        return
    print(
        f"  {label:<22s}"
        f"  mean={statistics.mean(samples_ms):7.2f}ms"
        f"  std={statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0:6.2f}ms"
        f"  min={min(samples_ms):7.2f}ms"
        f"  p50={statistics.median(samples_ms):7.2f}ms"
        f"  p95={np.percentile(samples_ms, 95):7.2f}ms"
        f"  max={max(samples_ms):7.2f}ms"
    )


def _build_uri(host: str, port: int | None) -> str:
    if host.startswith("https://"):
        return "wss://" + host[len("https://"):]
    if host.startswith("http://"):
        return "ws://" + host[len("http://"):]
    if host.startswith(("ws://", "wss://")):
        return host
    uri = f"ws://{host}"
    if port is not None:
        uri += f":{port}"
    return uri


def _connect(uri: str, api_key: str | None) -> websockets.sync.client.ClientConnection:
    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
    ws = websockets.sync.client.connect(uri, compression=None, max_size=None, additional_headers=headers)
    ws.recv()  # consume server metadata handshake
    return ws


def run_network_profile(args) -> None:
    """Connect to a remote WebSocket server and profile send/receive latency.

    With --batch-size N, opens N connections each with a distinct robot_id so
    the server scheduler receives N simultaneous requests and runs a true
    batch-N inference call.
    """
    uri = _build_uri(args.host, args.port)
    batch_size = args.batch_size

    print(f"Connecting {batch_size} client(s) to {uri} ...")
    connections = [
        (_connect(uri, args.api_key), f"profile_robot_{r}")
        for r in range(batch_size)
    ]
    print(f"Connected ({batch_size} robot{'s' if batch_size > 1 else ''}).\n")

    example = libero_policy.make_libero_example()
    total_iters = args.num_warmup + args.num_iters

    # Accumulators — one list per robot, indexed [robot][iter]
    t_rtt:           list[list[float]] = [[] for _ in range(batch_size)]
    t_server_compute: list[list[float]] = [[] for _ in range(batch_size)]
    t_network:       list[list[float]] = [[] for _ in range(batch_size)]
    t_send_dur:      list[list[float]] = [[] for _ in range(batch_size)]
    t_recv_dur:      list[list[float]] = [[] for _ in range(batch_size)]
    batch_rtts:      list[float] = []  # time from first send to last recv across all robots

    # Barrier so all threads send as close to simultaneously as possible.
    barrier = threading.Barrier(batch_size)

    for i in range(total_iters):
        is_warmup = i < args.num_warmup
        iter_results: list[dict] = [{}] * batch_size

        def _robot_iter(idx: int, ws, robot_id: str) -> None:
            request = InferRequest(
                robot_id=robot_id,
                observation=example,
                start_step=i * 10,
                request_timestamp=time.time(),
                deadline=float("inf"),
                infer_type=InferType.SYNC,
                params=None,
            )
            data = msgpack_numpy.packb(asdict(request))
            barrier.wait()  # synchronise all robots before sending
            t_send = time.perf_counter()
            ws.send(data)
            t_after_send = time.perf_counter()
            response_bytes = ws.recv()
            t_recv = time.perf_counter()
            response = msgpack_numpy.unpackb(response_bytes)
            iter_results[idx] = {
                "t_send": t_send,
                "t_after_send": t_after_send,
                "t_recv": t_recv,
                "server_ms": response.get("server_compute_ms", 0.0),
                "req_bytes": len(data),
                "resp_bytes": len(response_bytes),
            }

        threads = [
            threading.Thread(target=_robot_iter, args=(idx, ws, robot_id))
            for idx, (ws, robot_id) in enumerate(connections)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Batch RTT: from the earliest send to the latest recv across all robots.
        first_send = min(r["t_send"] for r in iter_results)
        last_recv  = max(r["t_recv"] for r in iter_results)
        batch_rtt_ms = (last_recv - first_send) * 1e3

        if not is_warmup:
            batch_rtts.append(batch_rtt_ms)
            for idx, r in enumerate(iter_results):
                rtt_ms    = (r["t_recv"] - r["t_send"]) * 1e3
                send_ms   = (r["t_after_send"] - r["t_send"]) * 1e3
                server_ms = r["server_ms"]
                recv_ms   = (r["t_recv"] - r["t_after_send"]) * 1e3
                t_rtt[idx].append(rtt_ms)
                t_server_compute[idx].append(server_ms)
                t_network[idx].append(rtt_ms - server_ms)
                t_send_dur[idx].append(send_ms)
                t_recv_dur[idx].append(recv_ms)

        if args.verbose:
            label = "warmup" if is_warmup else "timed "
            per_robot = "  ".join(
                f"r{idx}: rtt={(r['t_recv']-r['t_send'])*1e3:.0f}ms"
                f"(send={(r['t_after_send']-r['t_send'])*1e3:.0f}ms"
                f" infer={r['server_ms']:.0f}ms"
                f" recv={(r['t_recv']-r['t_after_send'])*1e3:.0f}ms)"
                for idx, r in enumerate(iter_results)
            )
            print(f"  [{label} {i:3d}] batch_rtt={batch_rtt_ms:.2f}ms  {per_robot}")

    for ws, _ in connections:
        ws.close()

    # Flatten per-robot lists for aggregate stats
    all_rtt     = [v for lst in t_rtt for v in lst]
    all_server  = [v for lst in t_server_compute for v in lst]
    all_network = [v for lst in t_network for v in lst]
    all_send    = [v for lst in t_send_dur for v in lst]
    all_recv    = [v for lst in t_recv_dur for v in lst]

    print(f"\n--- Latency profile: batch_size={batch_size}, {args.num_iters} iters ---")
    print(f"  Request payload:   {iter_results[0]['req_bytes'] / 1024:.1f} KB  (per robot)")
    print(f"  Response payload:  {iter_results[0]['resp_bytes'] / 1024:.1f} KB  (per robot)")
    print()
    if batch_size > 1:
        _print_stats("batch RTT (wall)", batch_rtts)
        print()
    _print_stats("per-robot RTT", all_rtt)
    _print_stats("  send", all_send)
    _print_stats("  server infer", all_server)
    _print_stats("  recv", all_recv)
    _print_stats("  network overhead", all_network)


def run_local(args) -> None:
    """Run inference locally (original behaviour)."""
    config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

    policy = policy_config.create_trained_policy(config, checkpoint_dir, sample_kwargs={"num_steps": args.num_steps})

    example = libero_policy.make_libero_example()
    request = InferRequest(robot_id="test_robot", observation=example, start_step=0, request_timestamp=0.0, deadline=float("inf"), infer_type=InferType.SYNC, params=None)
    requests = [request] * args.batch_size

    print("Warming up...")
    outputs = policy.infer_batch(requests)
    jax.block_until_ready(outputs)
    print("Warmed up")

    print("Inferring...")
    for _ in range(5):
        outputs = policy.infer_batch(requests)
        jax.block_until_ready(outputs)
    print("Inference complete")


def main(args):
    if args.host is not None:
        run_network_profile(args)
    else:
        run_local(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Local mode args
    parser.add_argument("--num-steps", type=int, default=10)
    # Shared
    parser.add_argument("--batch-size", type=int, default=1, help="Number of concurrent robot connections (network mode) or local batch size.")
    # Network profiling args
    parser.add_argument("--host", type=str, default=None, help="Remote server host. If set, runs network latency profiling instead of local inference.")
    parser.add_argument("--port", type=int, default=None, help="Remote server port.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for server authentication.")
    parser.add_argument("--num-iters", type=int, default=20, help="Number of timed iterations.")
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warmup iterations (excluded from stats).")
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration timings.")
    args = parser.parse_args()
    main(args)
