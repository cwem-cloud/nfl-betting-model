import pytest

from pocket_capper.models import distributions as dist


def test_pmf_mean_preserved_and_key_numbers():
    x, p = dist.margin_pmf(3.0, 13.2, "nfl")
    assert (x * p).sum() == pytest.approx(3.0, abs=0.1)
    # 3 is a key number: more mass than 2 or 4
    assert p[x == 3][0] > p[x == 4][0] and p[x == 3][0] > p[x == 2][0]


def test_spread_probs_push_on_key():
    pr = dist.spread_probs(3.0, 13.2, -3.0, "nfl")
    assert pr["push"] > 0.07  # empirically ~9% of games with a 3-pt spread land on exactly 3
    assert pr["home"] + pr["away"] + pr["push"] == pytest.approx(1.0)
    half = dist.spread_probs(3.0, 13.2, -2.5, "nfl")
    assert half["push"] == 0
    assert half["home"] > pr["home"]


def test_symmetry_and_win_prob():
    assert dist.win_prob(0.0, 13.2) == pytest.approx(0.5, abs=1e-6)
    assert dist.win_prob(7.0, 13.2) > 0.65
    t = dist.total_probs(45.0, 13.3, 45.5)
    assert t["over"] < 0.5 < t["over"] + t["under"] + 1e-9
    assert dist.margin_to_spread(3.2) == -3.0
