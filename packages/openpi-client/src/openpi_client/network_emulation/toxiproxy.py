from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
import math
import pathlib
import subprocess
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import requests


DEFAULT_TOXIC_UPSTREAM = "latency_upstream"
DEFAULT_TOXIC_DOWNSTREAM = "latency_downstream"


class NetworkEmulationConfigError(ValueError):
    """Raised when network emulation config is invalid."""


@dataclass(frozen=True)
class ToxiproxyConfig:
    api_url: str = "http://127.0.0.1:8474"
    listen_host: str = "127.0.0.1"
    listen_port_base: int = 18080
    server_args: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SamplingConfig:
    default_seed: int = 0
    resample_every_requests: int = 1


@dataclass(frozen=True)
class RobotLatencyConfig:
    rtt_median_ms: float
    rtt_sigma: float
    seed: Optional[int] = None


@dataclass(frozen=True)
class NetworkEmulationConfig:
    toxiproxy: ToxiproxyConfig
    sampling: SamplingConfig
    robots: Dict[str, RobotLatencyConfig]

    @classmethod
    def from_json(cls, path: Union[str, pathlib.Path]) -> "NetworkEmulationConfig":
        data = json.loads(pathlib.Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkEmulationConfig":
        if not isinstance(data, dict):
            raise NetworkEmulationConfigError("Network config must be a JSON object")

        toxi_data = data.get("toxiproxy") or {}
        sampling_data = data.get("sampling") or {}
        robots_data = data.get("robots")

        if not isinstance(robots_data, dict) or not robots_data:
            raise NetworkEmulationConfigError("network_config.robots must be a non-empty object")

        toxiproxy = ToxiproxyConfig(
            api_url=str(toxi_data.get("api_url", ToxiproxyConfig.api_url)),
            listen_host=str(toxi_data.get("listen_host", ToxiproxyConfig.listen_host)),
            listen_port_base=int(toxi_data.get("listen_port_base", ToxiproxyConfig.listen_port_base)),
            server_args=[str(x) for x in toxi_data.get("server_args", [])],
        )
        sampling = SamplingConfig(
            default_seed=int(sampling_data.get("default_seed", SamplingConfig.default_seed)),
            resample_every_requests=int(
                sampling_data.get("resample_every_requests", SamplingConfig.resample_every_requests)
            ),
        )

        robots: Dict[str, RobotLatencyConfig] = {}
        for robot_id, robot_cfg in robots_data.items():
            if not isinstance(robot_cfg, dict):
                raise NetworkEmulationConfigError(f"robot config for {robot_id!r} must be an object")
            if "rtt_median_ms" not in robot_cfg or "rtt_sigma" not in robot_cfg:
                raise NetworkEmulationConfigError(
                    f"{robot_id} must define rtt_median_ms and rtt_sigma (mean/std fields are no longer supported)"
                )
            robots[str(robot_id)] = RobotLatencyConfig(
                rtt_median_ms=float(robot_cfg["rtt_median_ms"]),
                rtt_sigma=float(robot_cfg["rtt_sigma"]),
                seed=int(robot_cfg["seed"]) if robot_cfg.get("seed") is not None else None,
            )

        config = cls(toxiproxy=toxiproxy, sampling=sampling, robots=robots)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.toxiproxy.api_url.startswith("http://"):
            raise NetworkEmulationConfigError("toxiproxy.api_url must start with http://")
        if not self.toxiproxy.listen_host:
            raise NetworkEmulationConfigError("toxiproxy.listen_host must be non-empty")
        if self.toxiproxy.listen_port_base <= 0:
            raise NetworkEmulationConfigError("toxiproxy.listen_port_base must be > 0")
        if self.sampling.resample_every_requests <= 0:
            raise NetworkEmulationConfigError("sampling.resample_every_requests must be >= 1")

        for robot_id, robot_cfg in self.robots.items():
            if robot_cfg.rtt_median_ms <= 0:
                raise NetworkEmulationConfigError(f"{robot_id}.rtt_median_ms must be > 0")
            if robot_cfg.rtt_sigma < 0:
                raise NetworkEmulationConfigError(f"{robot_id}.rtt_sigma must be >= 0")


@dataclass(frozen=True)
class WorkerNetworkContext:
    robot_id: str
    proxy_name: str
    proxy_host: str
    proxy_port: int
    api_url: str
    rtt_median_ms: float
    rtt_sigma: float
    seed: int
    resample_every_requests: int
    trace_path: str


@dataclass(frozen=True)
class LatencyTraceEntry:
    request_index: int
    sampled_rtt_ms: float
    upstream_latency_ms: int
    downstream_latency_ms: int
    resampled: bool
    timestamp: float


class LogNormalRttSampler:
    """Samples RTT (milliseconds) from LogNormal(log(median_ms), sigma)."""

    def __init__(self, median_ms: float, sigma: float, seed: int) -> None:
        if median_ms <= 0:
            raise ValueError("median_ms must be > 0")
        if sigma < 0:
            raise ValueError("sigma must be >= 0")

        self._median_ms = float(median_ms)
        self._sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

        self._mu = math.log(self._median_ms)

    def sample(self) -> float:
        if self._sigma == 0:
            return self._median_ms
        return float(self._rng.lognormal(self._mu, self._sigma))


class ToxiproxyHttpClient:
    """Minimal toxiproxy HTTP client with idempotent proxy/toxic helpers."""

    def __init__(self, api_url: str, *, session: Optional[requests.Session] = None, timeout_s: float = 2.0) -> None:
        self._api_url = api_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout_s = timeout_s

    def _url(self, path: str) -> str:
        return f"{self._api_url}{path}"

    def _request(self, method: str, path: str, *, expected: Tuple[int, ...], **kwargs) -> requests.Response:
        try:
            response = self._session.request(method, self._url(path), timeout=self._timeout_s, **kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(f"toxiproxy request failed: {method} {path}: {exc}") from exc

        if response.status_code not in expected:
            raise RuntimeError(
                f"toxiproxy request failed: {method} {path} -> {response.status_code}: {response.text[:400]}"
            )
        return response

    def wait_until_ready(self, timeout_s: float = 10.0, poll_interval_s: float = 0.1) -> None:
        deadline = time.time() + timeout_s
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                self._request("GET", "/proxies", expected=(200,))
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(poll_interval_s)

        raise TimeoutError(f"Timed out waiting for toxiproxy API at {self._api_url}: {last_error}")

    def create_proxy(self, name: str, listen: str, upstream: str) -> None:
        payload = {
            "name": name,
            "listen": listen,
            "upstream": upstream,
            "enabled": True,
        }
        response = self._request("POST", "/proxies", expected=(200, 201, 409), json=payload)
        if response.status_code == 409:
            self.delete_proxy(name)
            self._request("POST", "/proxies", expected=(200, 201), json=payload)

    def delete_proxy(self, name: str) -> None:
        self._request("DELETE", f"/proxies/{name}", expected=(200, 204, 404))

    def upsert_latency_toxic(
        self,
        proxy_name: str,
        toxic_name: str,
        stream: str,
        latency_ms: int,
        *,
        jitter_ms: int = 0,
        toxicity: float = 1.0,
    ) -> None:
        payload = {
            "name": toxic_name,
            "type": "latency",
            "stream": stream,
            "toxicity": float(toxicity),
            "attributes": {
                "latency": int(latency_ms),
                "jitter": int(jitter_ms),
            },
        }

        response = self._request(
            "POST",
            f"/proxies/{proxy_name}/toxics/{toxic_name}",
            expected=(200, 201, 404, 405),
            json=payload,
        )
        if response.status_code in (404, 405):
            self._request(
                "POST",
                f"/proxies/{proxy_name}/toxics",
                expected=(200, 201),
                json=payload,
            )


class LocalToxiproxyServer:
    """Starts/stops a local toxiproxy server process."""

    def __init__(self, server_bin: str, api_url: str, server_args: Optional[List[str]] = None) -> None:
        self._server_bin = server_bin
        self._server_args = list(server_args or [])
        self._client = ToxiproxyHttpClient(api_url)
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        bin_path = pathlib.Path(self._server_bin)
        if not bin_path.exists():
            raise FileNotFoundError(f"toxiproxy server binary not found: {bin_path}")

        # This mode owns lifecycle for this run and should not silently reuse a pre-existing server.
        try:
            self._client.wait_until_ready(timeout_s=0.25, poll_interval_s=0.05)
        except TimeoutError:
            pass
        else:
            raise RuntimeError("toxiproxy API is already reachable; refusing to reuse an existing server instance")

        cmd = [str(bin_path), *self._server_args]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            self._client.wait_until_ready(timeout_s=15.0, poll_interval_s=0.1)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None


class RobotNetworkHook:
    """Worker-local hook that updates toxics before each inference request."""

    def __init__(self, context: WorkerNetworkContext) -> None:
        self._context = context
        self._client = ToxiproxyHttpClient(context.api_url)
        self._sampler = LogNormalRttSampler(
            median_ms=context.rtt_median_ms,
            sigma=context.rtt_sigma,
            seed=context.seed,
        )
        self._request_index = 0
        self._resample_every = max(1, context.resample_every_requests)
        half = max(0, int(round(context.rtt_median_ms / 2.0)))
        self._last_upstream_ms = half
        self._last_downstream_ms = half
        self._trace: List[LatencyTraceEntry] = []
        self._flushed_count = 0

    def _apply_latency(self, upstream_ms: int, downstream_ms: int) -> None:
        self._client.upsert_latency_toxic(
            self._context.proxy_name,
            DEFAULT_TOXIC_UPSTREAM,
            "upstream",
            upstream_ms,
        )
        self._client.upsert_latency_toxic(
            self._context.proxy_name,
            DEFAULT_TOXIC_DOWNSTREAM,
            "downstream",
            downstream_ms,
        )

    def before_send(self) -> None:
        self._request_index += 1
        should_resample = self._request_index == 1 or ((self._request_index - 1) % self._resample_every == 0)
        sampled = float(self._last_upstream_ms + self._last_downstream_ms)

        if should_resample:
            sampled = self._sampler.sample()
            half = max(0, int(round(sampled / 2.0)))
            self._last_upstream_ms = half
            self._last_downstream_ms = half
            self._apply_latency(self._last_upstream_ms, self._last_downstream_ms)

        self._trace.append(
            LatencyTraceEntry(
                request_index=self._request_index,
                sampled_rtt_ms=sampled,
                upstream_latency_ms=self._last_upstream_ms,
                downstream_latency_ms=self._last_downstream_ms,
                resampled=should_resample,
                timestamp=time.time(),
            )
        )

    def flush_trace(self) -> None:
        if self._flushed_count >= len(self._trace):
            return

        path = pathlib.Path(self._context.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for entry in self._trace[self._flushed_count :]:
                handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        self._flushed_count = len(self._trace)

    def close(self) -> None:
        self.flush_trace()


class NetworkEmulationManager:
    """Main-process manager: starts server, provisions proxies, and builds worker contexts."""

    def __init__(
        self,
        config: NetworkEmulationConfig,
        *,
        toxiproxy_server_bin: str,
        upstream_host: str,
        upstream_port: int,
        worker_count: int,
        output_dir: Union[str, pathlib.Path],
    ) -> None:
        self._config = config
        self._upstream_host = upstream_host
        self._upstream_port = upstream_port
        self._worker_count = worker_count
        self._output_dir = pathlib.Path(output_dir)
        self._client = ToxiproxyHttpClient(config.toxiproxy.api_url)
        self._server = LocalToxiproxyServer(
            toxiproxy_server_bin,
            config.toxiproxy.api_url,
            server_args=config.toxiproxy.server_args,
        )
        self._worker_contexts: Dict[str, WorkerNetworkContext] = {}
        self._active_proxy_names: List[str] = []

    @property
    def worker_contexts(self) -> Dict[str, WorkerNetworkContext]:
        return dict(self._worker_contexts)

    def _build_worker_contexts(self) -> Dict[str, WorkerNetworkContext]:
        contexts: Dict[str, WorkerNetworkContext] = {}
        for idx in range(self._worker_count):
            robot_id = f"robot_{idx}"
            robot_cfg = self._config.robots.get(robot_id)
            if robot_cfg is None:
                raise NetworkEmulationConfigError(
                    f"Missing robots.{robot_id} in network config for worker_count={self._worker_count}"
                )
            seed = robot_cfg.seed if robot_cfg.seed is not None else self._config.sampling.default_seed + idx
            contexts[robot_id] = WorkerNetworkContext(
                robot_id=robot_id,
                proxy_name=f"openpi_{robot_id}_proxy",
                proxy_host=self._config.toxiproxy.listen_host,
                proxy_port=self._config.toxiproxy.listen_port_base + idx,
                api_url=self._config.toxiproxy.api_url,
                rtt_median_ms=robot_cfg.rtt_median_ms,
                rtt_sigma=robot_cfg.rtt_sigma,
                seed=int(seed),
                resample_every_requests=self._config.sampling.resample_every_requests,
                trace_path=str(self._output_dir / f"{robot_id}_latency_trace.jsonl"),
            )
        return contexts

    def _write_resolved_config(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "toxiproxy": asdict(self._config.toxiproxy),
            "sampling": asdict(self._config.sampling),
            "upstream": {
                "host": self._upstream_host,
                "port": self._upstream_port,
            },
            "worker_contexts": {robot_id: asdict(ctx) for robot_id, ctx in self._worker_contexts.items()},
        }
        (self._output_dir / "resolved_config.json").write_text(json.dumps(payload, indent=2))

    def start(self) -> Dict[str, WorkerNetworkContext]:
        if self._worker_count == 0:
            self._worker_contexts = {}
            self._write_resolved_config()
            return {}

        self._server.start()
        self._worker_contexts = self._build_worker_contexts()

        upstream = f"{self._upstream_host}:{self._upstream_port}"
        self._active_proxy_names = []

        for context in self._worker_contexts.values():
            listen = f"{context.proxy_host}:{context.proxy_port}"
            self._client.create_proxy(context.proxy_name, listen=listen, upstream=upstream)

            half = max(0, int(round(context.rtt_median_ms / 2.0)))
            self._client.upsert_latency_toxic(
                context.proxy_name,
                DEFAULT_TOXIC_UPSTREAM,
                "upstream",
                half,
            )
            self._client.upsert_latency_toxic(
                context.proxy_name,
                DEFAULT_TOXIC_DOWNSTREAM,
                "downstream",
                half,
            )
            self._active_proxy_names.append(context.proxy_name)

        self._write_resolved_config()
        return dict(self._worker_contexts)

    def close(self) -> None:
        for proxy_name in self._active_proxy_names:
            try:
                self._client.delete_proxy(proxy_name)
            except Exception:
                # Best-effort cleanup.
                pass
        self._active_proxy_names = []
        self._server.stop()


def load_network_emulation_config(path: Union[str, pathlib.Path]) -> NetworkEmulationConfig:
    return NetworkEmulationConfig.from_json(path)
