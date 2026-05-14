"""
Availability DSL parser for First Steps scheduler.

Supported syntax examples:
  "any"                   – no restrictions
  "Thu only"              – single day
  "Mon, Wed"              – multiple days
  "M-W"                   – day range
  "Mon after 2pm"         – day with time floor
  "Tue, Thu before noon"  – days with time ceiling
  "M-W after 2pm"         – day range with time floor
  "not Mon mornings"      – exclusion (negation)
  "any after 9am"         – all days with time floor
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Scheduling constants (minutes from midnight)
DAY_START = 8 * 60 + 30  # 08:30
DAY_END = 16 * 60  # 16:00

DAYS = ["Mon", "Tue", "Wed", "Thu"]
DAY_INDICES = {d: i for i, d in enumerate(DAYS)}

# Canonical day name maps
_DAY_ALIASES: dict[str, str] = {
    "monday": "Mon",
    "mon": "Mon",
    "m": "Mon",
    "tuesday": "Tue",
    "tue": "Tue",
    "tu": "Tue",
    "t": "Tue",
    "wednesday": "Wed",
    "wed": "Wed",
    "w": "Wed",
    "thursday": "Thu",
    "thu": "Thu",
    "th": "Thu",
    "r": "Thu",
}

_MORNING_END = 12 * 60  # noon
_AFTERNOON_START = 12 * 60  # noon


def _parse_time(token: str) -> Optional[int]:
    """Convert a time token to minutes from midnight.

    Accepts: "2pm", "noon", "9am", "13:30", "2:30pm".
    Returns None if unparseable.
    """
    token = token.strip().lower()
    if token == "noon":
        return 12 * 60
    if token == "midnight":
        return 0

    # e.g. "2pm", "2:30pm", "14:00"
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", token)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _resolve_day(token: str) -> Optional[str]:
    """Return canonical day name (Mon/Tue/Wed/Thu) or None."""
    return _DAY_ALIASES.get(token.strip().lower())


def _expand_day_range(start: str, end: str) -> list[str]:
    """Expand 'Mon-Wed' into ['Mon','Tue','Wed'], restricted to M–Th."""
    s = _resolve_day(start)
    e = _resolve_day(end)
    if s is None or e is None:
        return []
    si = DAY_INDICES.get(s, -1)
    ei = DAY_INDICES.get(e, -1)
    if si == -1 or ei == -1 or si > ei:
        return []
    return DAYS[si : ei + 1]


@dataclass
class TimeWindow:
    """A single (day, earliest_start, latest_start) triple in minutes-from-midnight.

    latest_start is the latest time a 60-min session can BEGIN and still finish by DAY_END.
    """

    day: str  # "Mon" | "Tue" | "Wed" | "Thu"
    earliest: int = DAY_START  # minutes from midnight
    latest: int = DAY_END - 60  # session must end by DAY_END


@dataclass
class AvailabilityResult:
    """Parsed availability for one client."""

    windows: list[TimeWindow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.windows) > 0

    def summary(self) -> str:
        if not self.is_valid:
            return f"INVALID: {'; '.join(self.errors)}"
        parts = []
        for w in self.windows:
            e = _fmt_time(w.earliest)
            l = _fmt_time(w.latest + 60)  # show session-end ceiling
            parts.append(f"{w.day} {e}–{l}")
        return ", ".join(parts)


def _fmt_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    period = "am" if h < 12 else "pm"
    if h == 0:
        h = 12
    elif h > 12:
        h -= 12
    return f"{h}:{m:02d}{period}" if m else f"{h}{period}"


def _build_windows(
    days: list[str],
    earliest: int,
    latest_end: int,
    excluded_days: Optional[set[str]] = None,
) -> list[TimeWindow]:
    """Produce TimeWindow list, clamped to [DAY_START, DAY_END-60]."""
    excluded_days = excluded_days or set()
    windows: list[TimeWindow] = []
    for day in days:
        if day in excluded_days:
            continue
        e = max(earliest, DAY_START)
        # latest_end is the latest time the session must *end*; start = end - 60
        l = min(latest_end - 60, DAY_END - 60)
        if e > l:
            continue  # window too narrow after clamping
        windows.append(TimeWindow(day=day, earliest=e, latest=l))
    return windows


def parse_availability(raw: str) -> AvailabilityResult:
    """Parse a free-text availability string into structured TimeWindow objects.

    Returns an AvailabilityResult with windows and/or errors.
    """
    result = AvailabilityResult(raw=raw)
    text = raw.strip()

    if not text:
        result.errors.append("Empty availability string")
        return result

    lower = text.lower()

    # ── "any" variants ──────────────────────────────────────────────────────
    if re.match(r"^any\s*$", lower):
        result.windows = _build_windows(DAYS, DAY_START, DAY_END)
        return result

    # ── Negation: "not Mon mornings" / "not Mon" ────────────────────────────
    neg_match = re.match(
        r"^not\s+([a-z]+)(?:\s+(mornings?|afternoons?|evenings?))?$", lower
    )
    if neg_match:
        excl_day = _resolve_day(neg_match.group(1))
        if excl_day is None:
            result.errors.append(f"Unrecognised day in exclusion: '{neg_match.group(1)}'")
            return result
        period = neg_match.group(2) or ""
        remaining = [d for d in DAYS if d != excl_day]

        if period.startswith("morning"):
            # exclude that day in the morning → still allow afternoon
            for day in DAYS:
                if day == excl_day:
                    result.windows += _build_windows([day], _AFTERNOON_START, DAY_END)
                else:
                    result.windows += _build_windows([day], DAY_START, DAY_END)
        elif period.startswith("afternoon"):
            for day in DAYS:
                if day == excl_day:
                    result.windows += _build_windows([day], DAY_START, _MORNING_END)
                else:
                    result.windows += _build_windows([day], DAY_START, DAY_END)
        else:
            result.windows = _build_windows(remaining, DAY_START, DAY_END)

        if not result.windows:
            result.errors.append("No feasible windows after applying exclusion")
        return result

    # ── Tokenise into segments separated by commas ──────────────────────────
    # Each segment may be:  "<days> [before|after <time>]"
    # where <days> is one of: "any", day-name, day-range (Mon-Thu), comma-list
    segments = [s.strip() for s in text.split(",")]

    # Re-join if a comma appeared inside a time ("1,000" — unlikely but safe)
    # More importantly handle "Tue, Thu before noon" as TWO day names with ONE modifier
    # Strategy: collect day tokens, then parse modifier from the last segment.

    all_days: list[str] = []
    earliest = DAY_START
    latest_end = DAY_END
    time_modified = False

    for seg in segments:
        seg_lower = seg.lower().strip()

        # Check for time modifier within this segment
        # Patterns: "<days> after <time>", "<days> before <time>", "any after <time>"
        mod_match = re.search(
            r"(?P<prefix>.*?)\s*(?P<rel>after|before|until|from)\s+(?P<time>\S+)\s*$",
            seg_lower,
        )
        if mod_match:
            prefix = mod_match.group("prefix").strip()
            rel = mod_match.group("rel")
            time_tok = mod_match.group("time")
            t = _parse_time(time_tok)
            if t is None:
                result.errors.append(f"Could not parse time '{time_tok}' in '{seg}'")
                return result
            if rel in ("after", "from"):
                earliest = max(earliest, t) if time_modified else t
            else:  # before / until
                latest_end = min(latest_end, t) if time_modified else t
            time_modified = True
            seg_lower = prefix

        seg_lower = seg_lower.strip()
        if not seg_lower:
            continue

        # Resolve day tokens in this segment
        # Could be: "any", single day name, or day range "Mon-Wed"
        if seg_lower in ("any", "all", "every day", "everyday"):
            all_days.extend(DAYS)
        elif re.match(r"^[a-z]+-[a-z]+$", seg_lower):
            # Day range
            parts = seg_lower.split("-")
            expanded = _expand_day_range(parts[0], parts[1])
            if not expanded:
                result.errors.append(f"Cannot expand day range '{seg_lower}'")
                return result
            all_days.extend(expanded)
        else:
            # Might be "thu only", "mon only", etc.
            seg_lower = re.sub(r"\s+only$", "", seg_lower).strip()

            # Handle "mornings" / "afternoons" as time qualifiers without explicit times
            period_match = re.search(r"\s+(mornings?|afternoons?)$", seg_lower)
            if period_match:
                period = period_match.group(1)
                seg_lower = seg_lower[: period_match.start()].strip()
                if period.startswith("morning") and not time_modified:
                    latest_end = _MORNING_END
                    time_modified = True
                elif period.startswith("afternoon") and not time_modified:
                    earliest = _AFTERNOON_START
                    time_modified = True

            day = _resolve_day(seg_lower)
            if day is None:
                result.errors.append(f"Unrecognised day token: '{seg_lower}'")
                return result
            all_days.append(day)

    # De-duplicate while preserving Mon–Thu order
    seen: set[str] = set()
    ordered: list[str] = []
    for d in DAYS:
        if d in all_days and d not in seen:
            ordered.append(d)
            seen.add(d)

    if not ordered:
        result.errors.append(f"No valid days found in '{raw}'")
        return result

    result.windows = _build_windows(ordered, earliest, latest_end)

    if not result.windows:
        result.errors.append(
            f"No feasible time windows after parsing '{raw}' "
            f"(earliest={_fmt_time(earliest)}, latest_end={_fmt_time(latest_end)})"
        )

    return result
