import pytest

from pocket_capper.market import odds_math as om


def test_conversions():
    assert om.american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert om.american_to_decimal(150) == 2.5
    assert om.american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert om.prob_to_american(0.5238) == -110
    assert om.prob_to_american(0.4) == 150
    assert om.fmt_american(120) == "+120" and om.fmt_american(-105) == "-105"


def test_devig_sums_to_one():
    p = om.devig([-110, -110])
    assert p == pytest.approx([0.5, 0.5])
    p = om.devig([-150, 130])
    assert sum(p) == pytest.approx(1.0) and p[0] > 0.55


def test_ev_and_kelly():
    assert om.expected_value(0.5, -110) == pytest.approx(-0.04545, abs=1e-4)
    assert om.expected_value(0.55, -110) > 0
    assert om.kelly_fraction(0.5, -110) == 0
    assert om.kelly_fraction(0.55, -110) == pytest.approx(0.055, abs=1e-3)
    # a push returns the stake: same win prob with push mass is better than with loss mass
    assert om.expected_value(0.5, -110, p_push=0.05) > om.expected_value(0.5, -110)


def test_units_bounds():
    units_cfg = {"kelly_fraction": 0.25, "unit_pct_bankroll": 1.0, "min_units": 0.5, "max_units": 3.0}
    assert om.size_units(0.5, -110, 0, units_cfg) == 0
    u = om.size_units(0.58, -110, 0, units_cfg)
    assert 0.5 <= u <= 3.0 and (u * 4) == int(u * 4)
    assert om.size_units(0.9, -110, 0, units_cfg) == 3.0
    assert om.profit_units(2, -110, "win") == pytest.approx(1.818, abs=1e-3)
    assert om.profit_units(2, 150, "loss") == -2
