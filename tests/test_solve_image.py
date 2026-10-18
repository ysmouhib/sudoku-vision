import pytest
import torch
from torch import nn

from grids import SIZE
from puzzles import WIKI_9, WIKI_9_SOLUTION
from solve_image import read_and_solve


class FixedLogits(nn.Module):
    """A stand-in classifier that returns the same 81 x 10 logits whatever the cells."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)
        self.unused = nn.Parameter(torch.zeros(1))  # predict_logits reads the device from the parameters

    def forward(self, x):
        return self.logits[:len(x)]


def confident_logits(grid, confidence=10.0):
    z = torch.zeros(81, 10)
    for k, v in enumerate(v for row in grid for v in row):
        z[k, v] = confidence
    return z


def test_a_correct_reading_is_solved():
    out = read_and_solve(torch.zeros(SIZE, SIZE), FixedLogits(confident_logits(WIKI_9)))
    assert out["status"] == "solved"
    assert out["reading"] == out["final"] == WIKI_9
    assert out["solution"] == WIKI_9_SOLUTION


def test_a_misread_clue_is_repaired():
    z = confident_logits(WIKI_9)
    z[0, 3], z[0, 5] = 10.0, 8.0  # the 5 at (0, 0) read as 3, a duplicate of (0, 1); 5 second most likely
    out = read_and_solve(torch.zeros(SIZE, SIZE), FixedLogits(z))
    assert out["reading"][0][0] == 3
    assert out["status"] == "solved"
    assert out["final"] == WIKI_9
    assert out["solution"] == WIKI_9_SOLUTION


def test_the_image_size_is_checked():
    with pytest.raises(ValueError):
        read_and_solve(torch.zeros(28, 28), FixedLogits(confident_logits(WIKI_9)))
