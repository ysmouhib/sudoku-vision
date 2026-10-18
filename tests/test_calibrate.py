import random

import torch

from calibrate import fit_temperature, fit_vector_scaling, reliability
from generate import make_puzzle
from repair import repair


def sample_labels(logits, generator):
    """Draw one label per row from softmax(logits)."""
    return torch.multinomial(logits.softmax(1), 1, generator=generator).flatten()


def test_fit_temperature_recovers_the_true_temperature():
    g = torch.Generator().manual_seed(0)
    z = 3 * torch.randn(20000, 10, generator=g)
    labels = sample_labels(z / 2.0, g)  # data generated at temperature 2
    assert abs(fit_temperature(z, labels) - 2.0) < 0.1


def test_vector_scaling_fits_at_least_as_well_as_temperature():
    g = torch.Generator().manual_seed(1)
    z = 3 * torch.randn(20000, 10, generator=g)
    labels = sample_labels(z / 2.0 + torch.linspace(-1, 1, 10), g)
    t = fit_temperature(z, labels)
    a, b = fit_vector_scaling(z, labels)
    nll = lambda zz: torch.nn.functional.cross_entropy(zz, labels).item()
    assert nll(z * a + b) <= nll(z / t) + 1e-4


def test_ece_is_small_for_calibrated_probabilities_and_large_for_overconfident_ones():
    g = torch.Generator().manual_seed(2)
    z = 3 * torch.randn(50000, 10, generator=g)
    labels = sample_labels(z, g)
    assert reliability(z.softmax(1), labels)[0] < 0.01
    assert reliability((3 * z).softmax(1), labels)[0] > 0.05


def test_temperature_scaling_does_not_change_the_repair():
    rng = random.Random(0)
    g = torch.Generator().manual_seed(3)
    puzzle = make_puzzle(0)["puzzle"]
    z = 2 * torch.randn(81, 10, generator=g)
    for k, v in enumerate(val for row in puzzle for val in row):
        z[k, v] += 6  # the true class usually wins...
    for k in rng.sample(range(81), 6):
        z[k, rng.randrange(10)] += 7  # ...but six cells get a strong rival
    reference = repair(torch.log_softmax(z.double(), 1).tolist())
    for t in (0.5, 1.7, 4.0):
        assert repair(torch.log_softmax((z / t).double(), 1).tolist()) == reference
