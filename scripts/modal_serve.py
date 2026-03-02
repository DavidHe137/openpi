"""Serve the FastAPI policy server on Modal.

Connect with:
  uv run scripts/infer.py --host https://....modal.run --num-iters 50 --verbose
"""

import logging
import pathlib
import subprocess
import threading
import time
import urllib.error
import urllib.request

import modal

log = logging.getLogger(__name__)

app = modal.App("openpi-serve")

GPU = "h100"
MAX_NUM_ROBOTS = 10
REGION = "us-east-1"
ENV_MODE = "LIBERO"
MAX_BATCH_SIZE = 4
PORT = 8080

REPO_ROOT = pathlib.Path(__file__).parent.parent


checkpoint_volume = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)
CHECKPOINT_VOLUME_PATH = "/checkpoints"

REQUIREMENTS_FILE = REPO_ROOT / "requirements-modal.txt"

_MODAL_EXCLUDE = [
    "torch",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "openpi",
    "openpi-client",
    "av",
]


# NOTE: this is necessary because Modal does not support uv workspaces, which are in the pyproject.toml
def generate_requirements() -> None:
    """Export a flat requirements.txt for Modal (excludes packages installed separately)."""
    cmd = [
        "uv",
        "export",
        "--no-hashes",
        "--no-dev",
        "--no-emit-workspace",
        *[arg for pkg in _MODAL_EXCLUDE for arg in ("--no-emit-package", pkg)],
        "-o",
        str(REQUIREMENTS_FILE),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    print(f"Written {REQUIREMENTS_FILE}")


if modal.is_local():
    generate_requirements()

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
    _base.pip_install("av==12.3.0", "pytest==8.3.4")
    .pip_install_from_requirements(str(REQUIREMENTS_FILE))
    .env(
        {
            "OPENPI_DATA_HOME": CHECKPOINT_VOLUME_PATH,
            "JAX_COMPILATION_CACHE_DIR": f"{CHECKPOINT_VOLUME_PATH}/.cache/jax_compilation",
            "TORCHINDUCTOR_CACHE_DIR": f"{CHECKPOINT_VOLUME_PATH}/.cache/torch_inductor",
            "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true",
            "GCLOUD_ANONYMOUS_ACCESS": "True",
            # "MUJOCO_GL": "egl", # FIXME: shouldn't be needed
            "JAX_PLATFORMS": "cuda",
            "TF_CPP_MIN_LOG_LEVEL": "2",  # to suppress warnings
            "ABSL_FLAGS_VERBOSITY": "0",
        }
    )
    .add_local_python_source("openpi", "openpi_client")
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
)


@app.cls(
    gpu=GPU,
    image=image,
    volumes={CHECKPOINT_VOLUME_PATH: checkpoint_volume},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region=[REGION],
)
@modal.concurrent(max_inputs=MAX_NUM_ROBOTS)
class ModalPolicyServer:
    @modal.enter(snap=True)
    def startup(self) -> None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting server")

        cmd = [
            "python",
            "/root/scripts/serve_policy.py",
            "--env",
            ENV_MODE,
            "--max-batch-size",
            str(MAX_BATCH_SIZE),
            "--port",
            str(PORT),
        ]

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def _stream_logs(proc: subprocess.Popen) -> None:
            for line in proc.stdout:
                logger.info(line.rstrip())

        threading.Thread(target=_stream_logs, args=(self.process,), daemon=True).start()

        # Block until server is ready so Modal snapshots a fully-loaded server.
        logger.info("Waiting for server to be ready...")
        while True:
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/metadata", timeout=5)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        logger.info("Server ready, snapshot will be taken now.")

    @modal.web_server(
        port=PORT,  # wrapped code must listen on this port
        startup_timeout=10 * 60,  # how long can server startup take?
    )
    def serve(self):
        pass

    @modal.exit()
    def teardown(self) -> None:
        """Clean up subprocesses on container exit."""
        self.process.terminate()
