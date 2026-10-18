import pytest
import torch

pytest.importorskip("onnxruntime")

from digits import make_model  # noqa: E402
from export_onnx import export, onnx_logits  # noqa: E402


def test_onnx_matches_pytorch_on_any_batch_size(tmp_path):
    torch.manual_seed(0)
    model = make_model("cnn").eval()
    path = tmp_path / "cnn.onnx"
    export(model, path)
    for n in (1, 81, 200):
        x = torch.rand(n, 1, 28, 28)
        with torch.no_grad():
            expected = model(x).numpy()
        assert abs(onnx_logits(path, x) - expected).max() < 1e-4
