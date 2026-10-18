import random

import torch

from digits import digit_pool
from generate import make_puzzle, random_full_grid, shuffle_symmetries
from grids import SIZE, cut_cells, draw_grid, grid_cells, puzzle_labels
from puzzles import WIKI_9_SOLUTION
from sudoku_sat import count_solutions, is_valid_solution


def test_symmetries_keep_a_grid_valid():
    rng = random.Random(0)
    for _ in range(20):
        grid = shuffle_symmetries(WIKI_9_SOLUTION, rng)
        assert is_valid_solution([[0] * 9 for _ in range(9)], grid)


def test_full_grids_are_valid_and_varied():
    grids = [random_full_grid(random.Random(s)) for s in range(5)]
    for g in grids:
        assert is_valid_solution([[0] * 9 for _ in range(9)], g)
    assert len({str(g) for g in grids}) == 5


def test_generated_puzzles_have_exactly_one_solution():
    for seed in range(5):
        p = make_puzzle(seed)
        assert count_solutions(p["puzzle"]) == 1
        assert is_valid_solution(p["puzzle"], p["solution"])


def test_generated_puzzles_are_minimal():
    p = make_puzzle(0)["puzzle"]
    for r in range(9):
        for c in range(9):
            if p[r][c]:
                fewer = [row[:] for row in p]
                fewer[r][c] = 0
                assert count_solutions(fewer) == 2


def test_stop_at_keeps_that_many_clues():
    p = make_puzzle(0, stop_at=30)["puzzle"]
    assert sum(v != 0 for row in p for v in row) == 30
    assert count_solutions(p) == 1


def test_puzzles_are_fixed_by_the_seed():
    assert make_puzzle(3) == make_puzzle(3)


def test_cut_cells_gives_back_the_pasted_cells():
    cells = torch.rand(81, 1, 28, 28)
    image = draw_grid(cells)
    assert image.shape == (SIZE, SIZE)
    assert torch.equal(cut_cells(image), cells)


def test_test_grids_use_only_mnist_test_digits():
    puzzles = [make_puzzle(seed) for seed in range(3)]
    cells, sources = grid_cells(puzzles, "test")
    test_images, test_labels = digit_pool("test")
    labels = puzzle_labels(puzzles)
    for i in range(3):
        for k in range(81):
            if labels[i, k]:
                assert torch.equal(cells[i, k], test_images[sources[i, k]])
                assert test_labels[sources[i, k]] == labels[i, k]
            else:
                assert sources[i, k] == -1
    train_images, _ = digit_pool("train")
    train_set = {img.numpy().tobytes() for img in train_images[:, 0]}
    digit_cells = cells[labels != 0][:, 0]
    assert sum(c.numpy().tobytes() in train_set for c in digit_cells) == 0
