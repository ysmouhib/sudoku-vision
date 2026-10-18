import math
import random

from pysat.examples.rc2 import RC2

from generate import make_puzzle
from puzzles import WIKI_9
from repair import (cost_offset, pair_cut, reading_cost, reading_from_model, repair, repair_formula, two_solutions,
                    weight, y)
from sudoku_sat import count_solutions


def log_probs_for(reading, confidence=0.9):
    """Log-probabilities that put `confidence` on each cell's reading and spread the rest evenly."""
    rest = math.log((1 - confidence) / 9)
    return [[math.log(confidence) if k == reading[r][c] else rest for k in range(10)]
            for r in range(9) for c in range(9)]


def copy(grid):
    return [row[:] for row in grid]


def test_reading_variables_follow_the_729_solution_variables():
    numbers = [y(r, c, k) for r in range(9) for c in range(9) for k in range(10)]
    assert sorted(numbers) == list(range(730, 1540))


def test_weight_is_minus_1000_log_p_and_at_least_1():
    assert weight(math.log(0.5)) == 693
    assert weight(math.log(0.99999)) == 1
    assert weight(0.0) == 1


def test_correct_reading_comes_back_unchanged():
    readings = repair(log_probs_for(WIKI_9))
    assert readings == [WIKI_9]


def test_same_digit_twice_in_a_row_is_fixed_when_the_truth_is_second():
    lp = log_probs_for(WIKI_9)
    lp[0] = [math.log(p) for p in [0.005, 0.005, 0.005, 0.6, 0.005, 0.35, 0.005, 0.005, 0.01, 0.01]]  # 5 read as 3
    readings = repair(lp)
    assert readings[0] == WIKI_9


def test_blank_read_as_a_digit_is_fixed():
    lp = log_probs_for(WIKI_9)
    lp[2] = [math.log(p) for p in [0.35, 0.6] + [0.05 / 8] * 8]  # cell (0, 2) is blank, read as 1
    assert repair(lp)[0] == WIKI_9


def test_maxsat_cost_is_the_sum_of_the_chosen_weights_minus_the_offset():
    lp = log_probs_for(WIKI_9)
    lp[0] = [math.log(p) for p in [0.005, 0.005, 0.005, 0.6, 0.005, 0.35, 0.005, 0.005, 0.01, 0.01]]
    with RC2(repair_formula(lp)) as rc2:
        rc2.compute()
        assert rc2.cost == reading_cost(WIKI_9, lp) - cost_offset(lp)
        assert rc2.cost == weight(math.log(0.35)) - weight(math.log(0.6))  # only cell (0, 0) pays


def satisfies(reading, clause):
    """Return True if a reading (with its implied y literals) satisfies a clause of positive y literals."""
    return any(y(r, c, reading[r][c]) == lit for lit in clause for r in range(9) for c in range(9))


def test_pair_cut_excludes_readings_both_solutions_satisfy_and_keeps_the_true_one():
    puzzle = make_puzzle(1)["puzzle"]
    r, c = next((r, c) for r in range(9) for c in range(9) if puzzle[r][c])
    blanked = copy(puzzle)
    blanked[r][c] = 0
    a, b = two_solutions(blanked)
    cut = pair_cut(a, b)
    assert not satisfies(blanked, cut)
    assert not satisfies([[0] * 9 for _ in range(9)], cut)  # the empty reading has every solution
    assert satisfies(puzzle, cut)


def blocking_loop(log_probs, max_tries=200):
    """The plain uniqueness loop: block each reading with several solutions, ask for the next."""
    with RC2(repair_formula(log_probs)) as rc2:
        for _ in range(max_tries):
            reading = reading_from_model(rc2.compute())
            if count_solutions(reading) == 1:
                return reading
            rc2.add_clause([-y(r, c, reading[r][c]) for r in range(9) for c in range(9)])


def test_pair_cut_loop_ends_on_the_same_reading_as_the_blocking_loop():
    rng = random.Random(0)
    for seed in range(6):
        puzzle = make_puzzle(seed)["puzzle"]
        lp = log_probs_for(puzzle, confidence=0.98)
        clues = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c]]
        for r, c in rng.sample(clues, 3):  # three doubtful clues: blank or another digit beats the truth
            probs = [0.002] * 10
            probs[puzzle[r][c]] = rng.uniform(0.2, 0.45)
            probs[rng.choice([k for k in range(10) if k != puzzle[r][c]])] = rng.uniform(0.5, 0.75)
            total = sum(probs)
            lp[9 * r + c] = [math.log(p / total) for p in probs]
        assert repair(lp)[-1] == blocking_loop(lp)


def test_uniqueness_loop_restores_a_clue_read_as_blank():
    p = make_puzzle(0)
    puzzle = p["puzzle"]
    r, c = next((r, c) for r in range(9) for c in range(9) if puzzle[r][c])
    lp = log_probs_for(puzzle)
    probs = [0.05 / 8] * 10
    probs[0], probs[puzzle[r][c]] = 0.6, 0.35  # blank most likely, the true digit second
    lp[9 * r + c] = [math.log(q) for q in probs]
    readings = repair(lp)
    blanked = copy(puzzle)
    blanked[r][c] = 0
    assert readings[0] == blanked  # MaxSAT alone accepts the blank: consistent, but several solutions
    assert readings[-1] == puzzle  # the loop forbids it and finds the true reading
