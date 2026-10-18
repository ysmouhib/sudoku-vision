"""Export the CNN (seed 0) to ONNX, and check that ONNX Runtime gives the same logits as PyTorch on the test cells."""
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

from digits import MODEL_DIR, cell_dataset, load_model, predict_logits

RESULTS = Path("results")
ONNX_PATH = MODEL_DIR / "cnn_seed0.onnx"
TOLERANCE = 1e-4  # largest |logit difference| accepted between PyTorch and ONNX Runtime (float32)


def export(model: nn.Module, path: Path) -> None:
    """Export a classifier of N x 1 x 28 x 28 cells (input 'cells') to N x 10 logits (output 'logits'), any N.

    Uses the TorchScript exporter: the default one (dynamo=True) relies on torch.export, which fails on Python 3.14
    with torch 2.10 ('wrapper_descriptor' object has no attribute '__annotate__').
    """
    torch.onnx.export(model.cpu().eval(), (torch.zeros(81, 1, 28, 28),), str(path), input_names=["cells"],
                      output_names=["logits"], dynamic_axes={"cells": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)


def onnx_logits(path: Path, images: torch.Tensor) -> np.ndarray:
    """Return the logits computed by ONNX Runtime (CPU) on images."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(["logits"], {"cells": images.numpy()})[0]


def main() -> None:
    """Export, then compare PyTorch and ONNX Runtime on every test cell and on batches of 1 and 81 cells."""
    model = load_model("cnn", 0)
    export(model, ONNX_PATH)
    images, labels = cell_dataset("test")
    z_torch = predict_logits(model, images).numpy()
    z_onnx = onnx_logits(ONNX_PATH, images)
    diff = float(np.abs(z_torch - z_onnx).max())
    same_argmax = float((z_torch.argmax(1) == z_onnx.argmax(1)).mean())
    small = [float(np.abs(onnx_logits(ONNX_PATH, images[:n]) - z_torch[:n]).max()) for n in (1, 81)]
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    grid = images[:81].numpy()
    start = time.perf_counter()
    for _ in range(100):
        session.run(["logits"], {"cells": grid})
    ms_per_grid = 10 * (time.perf_counter() - start)
    row = dict(cells=len(images), max_abs_logit_diff=diff, same_argmax=same_argmax,
               max_diff_batch_1=small[0], max_diff_batch_81=small[1],
               accuracy_onnx=float((z_onnx.argmax(1) == labels.numpy()).mean()),
               ms_per_grid_onnx_cpu=ms_per_grid, file_kb=ONNX_PATH.stat().st_size / 1024)
    pd.DataFrame([row]).to_csv(RESULTS / "onnx_check.csv", index=False)
    print(f"{ONNX_PATH}: {row['file_kb']:.0f} KB. On {len(images)} test cells: max |logit difference| {diff:.2e}, "
          f"same argmax {100 * same_argmax:.2f}%, accuracy {100 * row['accuracy_onnx']:.2f}%; batches of 1 and 81: "
          f"{small[0]:.2e}, {small[1]:.2e}; {ms_per_grid:.2f} ms per grid of 81 cells on the CPU")
    assert diff < TOLERANCE and same_argmax == 1.0 and max(small) < TOLERANCE, "ONNX and PyTorch disagree"


if __name__ == "__main__":
    main()
