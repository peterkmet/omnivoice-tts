import torch


def get_device() -> str:
    """Auto-detect the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
