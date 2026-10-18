from generate import make_puzzle
from puzzles import WIKI_9, WIKI_9_SOLUTION
from read_grids import classify, to_grid
from sudoku_sat import count_solutions, solve

import torch


def copy(grid):
    return [row[:] for row in grid]


def test_to_grid_reads_row_by_row():
    assert to_grid(torch.arange(81))[1][0] == 9


def test_perfect_reading():
    assert classify(copy(WIKI_9), WIKI_9, WIKI_9_SOLUTION) == "perfect"


def test_blank_read_as_wrong_digit_has_no_solution():
    reading = copy(WIKI_9)
    reading[0][2] = 1  # the solution has 4 there
    assert classify(reading, WIKI_9, WIKI_9_SOLUTION) == "no solution"


def test_blank_read_as_its_solution_digit_is_harmless():
    reading = copy(WIKI_9)
    reading[0][2] = 4
    assert classify(reading, WIKI_9, WIKI_9_SOLUTION) == "unique, right"


def test_clue_read_as_blank_in_a_minimal_puzzle_gives_several():
    p = make_puzzle(0)
    reading = copy(p["puzzle"])
    r, c = next((r, c) for r in range(9) for c in range(9) if reading[r][c])
    reading[r][c] = 0
    assert classify(reading, p["puzzle"], p["solution"]) == "several"


def test_some_clue_substitution_gives_a_unique_wrong_solution():
    for seed in range(20):
        p = make_puzzle(seed)
        for r in range(9):
            for c in range(9):
                for d in range(1, 10):
                    if p["puzzle"][r][c] and d != p["puzzle"][r][c]:
                        reading = copy(p["puzzle"])
                        reading[r][c] = d
                        if count_solutions(reading) == 1:
                            assert solve(reading) != p["solution"]
                            assert classify(reading, p["puzzle"], p["solution"]) == "unique, wrong"
                            return
    raise AssertionError("no undetectable substitution found in 20 puzzles")
