"""
Modal deployment configuration for Krea Realtime 14B video generation server.
This script deploys the real-time video generation server on Modal with B200 GPU.
"""

import modal

# Create Modal app
app = modal.App("krea-realtime-video")

# CUDA version compatible with B200 and Modal's host driver
cuda_version = "12.8.1"
flavor = "devel"  # includes full CUDA toolkit needed for flash_attn compilation
operating_sys = "ubuntu22.04"
cuda_tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Define the Modal image with all dependencies
image = (
    # Start with NVIDIA CUDA image that includes full CUDA toolkit for compiling flash_attn
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.11")
    .entrypoint([])  # remove verbose logging by base image
    .apt_install(
        "ffmpeg",
        "git",
        "build-essential",
    )
    # Use uv_sync to install dependencies from pyproject.toml
    .uv_sync(
        uv_project_dir=".",
        frozen=True,  # Use uv.lock file
        gpu="b200",  # Enable GPU support for package resolution
    )
    # Install hf_transfer for 100x faster HuggingFace downloads
    .pip_install("hf-transfer")
    # Install flash_attn with GPU access during image build (needs nvcc from CUDA toolkit)
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
        gpu="b200",  # Compile with GPU access
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Enable fast downloads
    .run_commands(
        # Download 1.3B base model (~2GB, needed for VAE and text encoder)
        "huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir /root/wan_models/Wan2.1-T2V-1.3B",
        gpu="b200",
    )
    .run_commands(
        # Download full 14B base model (~56GB, cached by Modal for future builds)
        "huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir-use-symlinks False --local-dir /root/wan_models/Wan2.1-T2V-14B",
        gpu="b200",
    )
    .run_commands(
        # Download Krea Realtime model checkpoint (28GB, contains distilled weights)
        "huggingface-cli download krea/krea-realtime-video krea-realtime-video-14b.safetensors --local-dir /root/checkpoints",
        gpu="b200",
    )
    # Add all local source code (must be last to avoid rebuilding on code changes)
    .add_local_dir(
        ".",
        remote_path="/root/app",
        copy=True,  # Force copy files instead of using cache
        ignore=["__pycache__", ".git", "*.pyc", "outputs", "wan_models", "checkpoints", ".venv", "uv.lock", "*.lock"]
    )
)

@app.function(
    image=image,
    gpu="b200",
    timeout=3600,  # 1 hour timeout
    scaledown_window=300,  # 5 minutes idle timeout
    secrets=[],  # Add any required secrets here
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=600)
def web():
    """
    Web server endpoint that hosts the real-time video generation server.

    The server will be accessible at the Modal URL provided after deployment.
    Access the web UI at the root path (/).
    """
    import os

    # Set environment variables
    os.environ["MODEL_FOLDER"] = "/root/wan_models"
    os.environ["CONFIG"] = "/root/app/configs/self_forcing_server_14b.yaml"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["DO_COMPILE"] = "true"

    # Change to app directory
    os.chdir("/root/app")

    # Create symlink for checkpoints so relative path works
    if not os.path.exists("checkpoints"):
        os.symlink("/root/checkpoints", "checkpoints")
    # Start uvicorn server using subprocess.Popen
    import subprocess

    cmd = [
        "uvicorn",
        "release_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--log-level",
        "info",
    ]
    # Launch the server process and wait for it to complete
    subprocess.Popen(cmd)

