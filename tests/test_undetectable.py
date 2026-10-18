import math

import pandas as pd

from generate import make_puzzle
from repair import repair
from sudoku_sat import count_solutions
from undetectable import choose_tau, clue_fibers, margin, peers


def test_peers_are_twenty_cells():
    assert len(peers(4, 4)) == 20 and (4, 4) not in peers(4, 4)
    assert (0, 8) in peers(0, 0) and (2, 2) in peers(0, 0) and (3, 3) not in peers(0, 0)


def test_fiber_lemma_matches_brute_force():
    for seed in (0, 1):
        puzzle = make_puzzle(seed)["puzzle"]
        for f in clue_fibers(puzzle):
            assert f["m"] >= 2  # the puzzle is minimal
            for d in range(1, 10):
                if d != f["digit"]:
                    substituted = [row[:] for row in puzzle]
                    substituted[f["r"]][f["c"]] = d
                    assert min(2, count_solutions(substituted)) == f[f"fiber{d}"]


def test_fibers_of_a_non_minimal_puzzle_include_redundant_clues():
    puzzle = make_puzzle(2, stop_at=40)["puzzle"]
    fibers = clue_fibers(puzzle)
    assert any(f["m"] == 1 for f in fibers)  # removing a redundant clue keeps the solution unique


def confident_log_probs(puzzle, doubtful, confidence=0.999):
    """Log-probabilities sure of every cell, except `doubtful`: {cell index: {class: probability}}."""
    rows = []
    for k, v in enumerate(val for row in puzzle for val in row):
        probs = doubtful.get(k) or {v: confidence}
        rest = (1 - sum(probs.values())) / (10 - len(probs))
        rows.append([math.log(probs.get(j, rest)) for j in range(10)])
    return rows


def test_margin_of_an_undetectable_misreading_is_the_cell_doubt():
    # Find a clue whose substitution gives a unique wrong solution, then read it wrongly with some doubt.
    p = make_puzzle(0)
    puzzle, solution = p["puzzle"], p["solution"]
    f = next(f for f in clue_fibers(puzzle) if any(f[f"fiber{d}"] == 1 for d in range(1, 10) if d != f["digit"]))
    wrong = next(d for d in range(1, 10) if d != f["digit"] and f[f"fiber{d}"] == 1)
    k = 9 * f["r"] + f["c"]
    lp = confident_log_probs(puzzle, {k: {wrong: 0.7, f["digit"]: 0.29}})
    result = margin(lp, solution)
    assert result["answer"] == "wrong"  # the misreading is consistent and unique: undetected by logic alone
    assert result["rival_is_truth"]  # the best competing answer is the true grid...
    assert abs(result["margin"] - math.log(0.7 / 0.29)) < 0.01  # ...at the price of that one cell's doubt
    assert result["next_flag"]


def test_choose_tau_minimizes_ten_misses_plus_flags():
    table = pd.DataFrame(dict(answer=["wrong", "wrong", "right", "right", "right", "timeout"],
                              margin=[0.5, 3.0, 1.0, 5.0, 8.0, float("nan")]))
    # tau <= 0.5: cost 20; 0.5 < tau <= 1: 10; 1 < tau <= 3: 10 + 1 = 11; 3 < tau <= 5: 1; above 5: 2 or more
    assert choose_tau(table) == 3.05


def test_margin_of_a_confident_right_answer_is_large():
    p = make_puzzle(1)
    lp = confident_log_probs(p["puzzle"], {})
    assert repair(lp)[-1] == p["puzzle"]
    result = margin(lp, p["solution"])
    assert result["answer"] == "right" and result["margin"] > math.log(99)
