"""Serve the OpenPI policy on Modal with a direct WebSocket tunnel.

Usage:
  uv run modal run scripts/modal_serve.py::serve

It prints a tunnel URL like:
  Tunnel: wss://...modal.run
Then connect with:
  uv run scripts/infer.py --host https://....modal.run --num-iters 50 --verbose

Keep this terminal open — Ctrl-C stops the server.
"""

import dataclasses
import pathlib
import subprocess
import tempfile

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "openpi-serve"
GPU = "h100"
POLICY_PORT = 8080
NUM_STEPS = 10
MAX_BATCH_SIZE = 4
CONFIG_NAME = "pi05_libero"
CHECKPOINT_URL = "gs://openpi-assets/checkpoints/pi05_libero"

REPO_ROOT = pathlib.Path(__file__).parent.parent

app = modal.App(APP_NAME)

checkpoint_volume = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)
CHECKPOINT_VOLUME_PATH = "/checkpoints"

# ---------------------------------------------------------------------------
# Requirements from uv lockfile
# ---------------------------------------------------------------------------
_LOCKFILE_EXCLUDE = [
    "torch", "jax", "jaxlib",
    "jax-cuda12-plugin", "jax-cuda12-pjrt",
    "openpi", "openpi-client",
    "av",
]


def _export_requirements() -> pathlib.Path:
    import shutil
    if not shutil.which("uv"):
        return pathlib.Path("/dev/null")
    cmd = [
        "uv", "export",
        "--no-hashes", "--no-dev", "--no-emit-workspace",
        *[arg for pkg in _LOCKFILE_EXCLUDE for arg in ("--no-emit-package", pkg)],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    req_file = pathlib.Path(tempfile.mktemp(suffix="-openpi-modal.txt"))
    req_file.write_text(result.stdout)
    return req_file


_req_file = _export_requirements()

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libglfw3", "libosmesa6", "libegl1")
    .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("jax[cuda12]==0.5.3", find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
)

image = (
    _base
    .pip_install("av==12.3.0", "pytest==8.3.4")
    .pip_install_from_requirements(str(_req_file))
    .add_local_python_source("openpi", "openpi_client")
)

# ---------------------------------------------------------------------------
# Server — run with: modal run scripts/modal_serve.py::serve
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU,
    image=image,
    volumes={CHECKPOINT_VOLUME_PATH: checkpoint_volume},
    timeout=3600,
    memory=32768,
)
def serve() -> None:
    import logging
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["OPENPI_DATA_HOME"] = CHECKPOINT_VOLUME_PATH
    os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"
    os.environ["GCLOUD_ANONYMOUS_ACCESS"] = "True"
    os.environ["MUJOCO_GL"] = "egl"

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.shared import download
    from openpi.training import config as _config
    from openpi_client.schemas import ServerMetadata

    config = _config.get_config(CONFIG_NAME)
    checkpoint_dir = download.maybe_download(CHECKPOINT_URL)
    checkpoint_volume.commit()

    def policy_factory():
        policy = _policy_config.create_trained_policy(
            config, checkpoint_dir, sample_kwargs={"num_steps": NUM_STEPS}, batch_size=MAX_BATCH_SIZE
        )
        if "env" not in policy._metadata:  # noqa: SLF001
            policy._metadata["env"] = "libero"  # noqa: SLF001
        return policy

    server_metadata = ServerMetadata(
        config_name=config.name,
        checkpoint_dir=str(checkpoint_dir),
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
        num_steps=NUM_STEPS,
        max_batch_size=MAX_BATCH_SIZE,
        env="libero",
        policy_metadata=config.policy_metadata or {},
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy_factory=policy_factory,
        host="0.0.0.0",
        port=POLICY_PORT,
        metadata=dataclasses.asdict(server_metadata),
        max_batch_size=MAX_BATCH_SIZE,
    )

    with modal.forward(POLICY_PORT) as tunnel:
        logging.info("Tunnel: %s", tunnel.url)
        print(f"\nTunnel: {tunnel.url}\n")
        server.serve_forever()