"""Tests for the availability DSL parser."""

import pytest

from scheduler.parser import (
    DAY_END,
    DAY_START,
    DAYS,
    parse_availability,
)

SESSION = 60  # minutes


# ── Helpers ──────────────────────────────────────────────────────────────────


def day_names(result) -> list[str]:
    return [w.day for w in result.windows]


def unique_days(result) -> set[str]:
    return set(day_names(result))


# ── "any" ────────────────────────────────────────────────────────────────────


def test_any_covers_all_days():
    r = parse_availability("any")
    assert r.is_valid
    assert unique_days(r) == {"Mon", "Tue", "Wed", "Thu"}


def test_any_windows_span_full_day():
    r = parse_availability("any")
    for w in r.windows:
        assert w.earliest == DAY_START
        assert w.latest == DAY_END - SESSION


# ── Single day ───────────────────────────────────────────────────────────────


def test_single_day_thu_only():
    r = parse_availability("Thu only")
    assert r.is_valid
    assert day_names(r) == ["Thu"]


def test_single_day_monday():
    r = parse_availability("Mon")
    assert r.is_valid
    assert day_names(r) == ["Mon"]


def test_single_day_tuesday_alias():
    r = parse_availability("Tuesday")
    assert r.is_valid
    assert day_names(r) == ["Tue"]


# ── Multiple days ─────────────────────────────────────────────────────────────


def test_multiple_days_comma():
    r = parse_availability("Tue, Thu")
    assert r.is_valid
    assert unique_days(r) == {"Tue", "Thu"}


def test_multiple_days_with_time_ceiling():
    r = parse_availability("Tue, Thu before noon")
    assert r.is_valid
    assert unique_days(r) == {"Tue", "Thu"}
    for w in r.windows:
        # latest start must be such that session ends by noon (12:00 = 720)
        assert w.latest <= 720 - SESSION


# ── Day ranges ───────────────────────────────────────────────────────────────


def test_day_range_m_to_w():
    r = parse_availability("M-W")
    assert r.is_valid
    assert unique_days(r) == {"Mon", "Tue", "Wed"}


def test_day_range_mon_to_thu():
    r = parse_availability("Mon-Thu")
    assert r.is_valid
    assert unique_days(r) == {"Mon", "Tue", "Wed", "Thu"}


def test_day_range_with_time_floor():
    r = parse_availability("M-W after 2pm")
    assert r.is_valid
    for w in r.windows:
        assert w.earliest == 14 * 60  # 14:00


# ── Time modifiers ────────────────────────────────────────────────────────────


def test_after_time_floor():
    r = parse_availability("Mon after 2pm")
    assert r.is_valid
    w = r.windows[0]
    assert w.earliest == 14 * 60


def test_before_time_ceiling():
    r = parse_availability("Mon before noon")
    assert r.is_valid
    w = r.windows[0]
    assert w.latest <= 12 * 60 - SESSION


def test_after_9am():
    r = parse_availability("any after 9am")
    assert r.is_valid
    for w in r.windows:
        assert w.earliest == 9 * 60


def test_noon_token():
    r = parse_availability("Thu before noon")
    assert r.is_valid
    w = r.windows[0]
    assert w.latest <= 12 * 60 - SESSION


def test_time_with_minutes():
    r = parse_availability("Mon after 1:30pm")
    assert r.is_valid
    w = r.windows[0]
    assert w.earliest == 13 * 60 + 30


# ── Negation ──────────────────────────────────────────────────────────────────


def test_not_mon():
    r = parse_availability("not Mon")
    assert r.is_valid
    assert "Mon" not in unique_days(r)
    assert unique_days(r) == {"Tue", "Wed", "Thu"}


def test_not_mon_mornings():
    r = parse_availability("not Mon mornings")
    assert r.is_valid
    # Mon should still appear but with afternoon-only window
    mon_windows = [w for w in r.windows if w.day == "Mon"]
    assert mon_windows, "Mon should still be available in afternoon"
    for w in mon_windows:
        assert w.earliest >= 12 * 60, "Mon should only be available after noon"


def test_not_thursday():
    r = parse_availability("not Thursday")
    assert r.is_valid
    assert "Thu" not in unique_days(r)


# ── Period keywords ───────────────────────────────────────────────────────────


def test_mon_mornings():
    r = parse_availability("Mon mornings")
    assert r.is_valid
    w = r.windows[0]
    assert w.latest <= 12 * 60 - SESSION


def test_thu_afternoons():
    r = parse_availability("Thu afternoons")
    assert r.is_valid
    w = r.windows[0]
    assert w.earliest >= 12 * 60


# ── Clamping ──────────────────────────────────────────────────────────────────


def test_clamp_before_day_start():
    # "after 8am" should be clamped to DAY_START (8:30)
    r = parse_availability("any after 8am")
    assert r.is_valid
    for w in r.windows:
        assert w.earliest == DAY_START


def test_window_too_narrow_returns_error():
    # "Mon before 8am" is completely outside working hours
    r = parse_availability("Mon before 8am")
    assert not r.is_valid or len(r.windows) == 0


# ── Error cases ───────────────────────────────────────────────────────────────


def test_empty_string():
    r = parse_availability("")
    assert not r.is_valid
    assert r.errors


def test_unrecognised_day():
    r = parse_availability("Saturday")
    assert not r.is_valid


def test_summary_valid():
    r = parse_availability("any")
    assert "Mon" in r.summary()


def test_summary_invalid():
    r = parse_availability("Saturday")
    assert "INVALID" in r.summary()
