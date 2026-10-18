"""Check that PyTorch sees the GPU, then compare CPU and GPU on a matrix product."""
import time

import torch

N = 4000
FLOPS = 2 * N**3  # an N x N matrix product needs N^3 multiplications and N^3 additions


def time_matmul(device, repeats=5):
    """Return the average time of one N x N matrix product on `device`."""
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    a @ b  # warm-up: the first call loads libraries and is much slower
    if device == "cuda":
        torch.cuda.synchronize()  # wait until the GPU has really finished
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / repeats
    print(f"{device:>4}: {seconds * 1000:8.1f} ms per product, {FLOPS / seconds / 1e12:6.2f} TFLOP/s")
    return seconds


print("PyTorch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

cpu_seconds = time_matmul("cpu")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print(f"Memory: {props.total_memory / 2**30:.1f} GiB")
    print("bfloat16 supported:", torch.cuda.is_bf16_supported())
    gpu_seconds = time_matmul("cuda")
    print(f"GPU speed-up: {cpu_seconds / gpu_seconds:.0f}x")
elif torch.version.cuda is None:
    print("This is a CPU-only build of PyTorch: reinstall it with the CUDA command.")
else:
    print("This build supports CUDA, but no GPU was found: check the NVIDIA driver.")
