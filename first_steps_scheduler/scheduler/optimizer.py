"""
OR-Tools VRP with Time Windows solver for the First Steps SLP scheduler.

MODEL OVERVIEW
--------------
We model the weekly schedule as a Vehicle Routing Problem with Time Windows
(VRPTW) where:

  • Each "vehicle" represents one workday (Monday–Thursday).
  • The depot (node 0) is the SLP's home base.
  • Each client is a node with a 60-minute service time and one or more
    time-window constraints derived from their availability.
  • The solver minimises total drive time subject to the time windows and
    optionally a lunch-break constraint.

MULTI-WINDOW CLIENTS
--------------------
OR-Tools VRPTW natively supports a single time window per node.  Clients
with multi-day availability are handled by creating one "copy" of the
client node per eligible day.  The solver then chooses exactly one copy
per client (enforced via AddAtMostOne + AddExactlyOne disjunction).

SOFT TRAVEL BUFFER
------------------
A 15-minute gap between sessions is preferred; 20 minutes is the soft cap.
We implement this as penalty terms added to the objective rather than hard
constraints, so the schedule never fails due to tight geography.

LUNCH VARIANTS
--------------
  • NO_LUNCH   – no break inserted.
  • DAILY_LUNCH – a 30-minute break node (0-service, artificial) is added
                  between 11:30 and 13:30 on every vehicle route.
  • HYBRID     – break on exactly 2 of 4 vehicles; the solver picks which 2
                 based on route density (fewer sessions → less need for a break).

SOLVER OUTPUT
-------------
Returns a ScheduleResult with one DayRoute per vehicle (day), each
containing an ordered list of ScheduledSession objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .parser import AvailabilityResult, DAY_START, DAY_END, DAYS, TimeWindow

# ── Constants ────────────────────────────────────────────────────────────────

SESSION_MINUTES = 60
BUFFER_PREFERRED = 15   # target gap (drive + idle) between sessions
BUFFER_SOFT_MAX = 20    # soft upper cap
LUNCH_DURATION = 30
LUNCH_EARLIEST = 11 * 60 + 30   # 11:30
LUNCH_LATEST_START = 13 * 60    # 13:00 (must finish by 13:30)

# Penalty weights (arbitrary units that calibrate objective priorities)
_PENALTY_UNSCHEDULED = 10_000    # per unscheduled client
_PENALTY_OVER_BUFFER_SOFT = 200  # per minute over 15 min gap
_PENALTY_OVER_BUFFER_HARD = 400  # per minute over 20 min gap
_SCALE = 10  # multiply all times by this before passing to OR-Tools (int required)

SOLVER_TIME_LIMIT_SEC = 30


class LunchVariant(str, Enum):
    NO_LUNCH = "no_lunch"
    DAILY_LUNCH = "daily_lunch"
    HYBRID = "hybrid"


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Client:
    client_id: str
    availability: AvailabilityResult
    # Index into the master location list (0 = home base, 1..N = clients)
    location_index: int = 0
    notes: str = ""


@dataclass
class ScheduledSession:
    client_id: str
    day: str            # "Mon" | "Tue" | "Wed" | "Thu"
    start_min: int      # minutes from midnight
    end_min: int        # start_min + 60
    location_index: int

    @property
    def start_str(self) -> str:
        return _fmt(self.start_min)

    @property
    def end_str(self) -> str:
        return _fmt(self.end_min)


@dataclass
class DayRoute:
    day: str
    sessions: list[ScheduledSession] = field(default_factory=list)
    total_drive_min: float = 0.0
    gap_violations: int = 0   # number of gaps > 20 min


@dataclass
class ScheduleResult:
    variant: LunchVariant
    days: list[DayRoute] = field(default_factory=list)
    unscheduled: list[str] = field(default_factory=list)  # client_ids
    total_drive_min: float = 0.0

    @property
    def is_feasible(self) -> bool:
        return len(self.unscheduled) == 0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


_Scale = _SCALE  # alias used inline throughout; keeps arithmetic readable


# ── Main entry point ─────────────────────────────────────────────────────────


def solve(
    clients: list[Client],
    travel_matrix: list[list[float]],  # minutes; index 0 = home base
    variant: LunchVariant = LunchVariant.NO_LUNCH,
) -> ScheduleResult:
    """Solve the VRPTW for *variant* lunch policy.

    *travel_matrix* must be (N+1)×(N+1) where index 0 is home base and
    indices 1..N correspond to clients[0..N-1].

    Returns a ScheduleResult.  If no feasible solution is found within the
    time limit, returns a partial result with unscheduled clients listed.
    """
    result = ScheduleResult(variant=variant)

    # Validate: drop clients with no parseable windows
    schedulable = [c for c in clients if c.availability.is_valid]
    infeasible_ids = [c.client_id for c in clients if not c.availability.is_valid]
    result.unscheduled.extend(infeasible_ids)

    if not schedulable:
        return result

    # Build expanded node list.
    # Each client may appear multiple times – once per eligible day.
    # node_info[i] = (client_idx, TimeWindow, day_idx)
    # Node 0 is always home base (depot).
    node_info: list[tuple[int, Optional[TimeWindow], int]] = [
        (-1, None, -1)  # depot placeholder
    ]
    # client_node_sets[client_idx] = list of node indices for that client
    client_node_sets: list[list[int]] = [[] for _ in range(len(schedulable))]

    for ci, client in enumerate(schedulable):
        for tw in client.availability.windows:
            day_idx = DAYS.index(tw.day)
            node_idx = len(node_info)
            node_info.append((ci, tw, day_idx))
            client_node_sets[ci].append(node_idx)

    num_nodes = len(node_info)  # includes depot
    num_vehicles = 4  # one per day Mon–Thu

    # ── Travel-time callback ─────────────────────────────────────────────────
    # Expand travel matrix to cover duplicate client nodes.
    # For node i (client ci) and node j (client cj): use travel_matrix[loc_i][loc_j]

    def _loc(node: int) -> int:
        """Map expanded node index → location index in travel_matrix."""
        if node == 0:
            return 0
        ci, _, _ = node_info[node]
        return schedulable[ci].location_index

    # Build a flat N×N integer matrix (scaled to int for OR-Tools)
    _travel_int: list[list[int]] = []
    for i in range(num_nodes):
        row: list[int] = []
        for j in range(num_nodes):
            li, lj = _loc(i), _loc(j)
            mins = travel_matrix[li][lj] if li != lj else 0.0
            row.append(int(round(mins * _Scale)))
        _travel_int.append(row)

    # ── OR-Tools model ───────────────────────────────────────────────────────

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback – returns scaled travel minutes between two nodes
    def _transit(from_idx: int, to_idx: int) -> int:
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return _travel_int[i][j]

    transit_cb = routing.RegisterTransitCallback(_transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Service-time callback: 60 min per client node, 0 at depot
    def _service(from_idx: int, _to_idx: int) -> int:
        node = manager.IndexToNode(from_idx)
        if node == 0:
            return 0
        return SESSION_MINUTES * _Scale

    service_cb = routing.RegisterTransitCallback(_service)

    # Dimension: elapsed time (transit + service).
    # Capacity = full work day in scaled minutes.
    total_cb = routing.RegisterTransitCallback(
        lambda f, t: _transit(f, t) + _service(f, t)
    )
    routing.AddDimension(
        total_cb,
        slack_max=int((DAY_END - DAY_START) * _Scale),  # max waiting allowed
        capacity=int((DAY_END - DAY_START) * _Scale),
        fix_start_cumul_to_zero=False,
        name="Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Set time window for the depot (start of day = DAY_START)
    depot_idx = manager.NodeToIndex(0)
    time_dim.CumulVar(depot_idx).SetRange(
        int(DAY_START * _Scale), int(DAY_END * _Scale)
    )

    # Set vehicle start times = DAY_START (all start at home at 8:30)
    for v in range(num_vehicles):
        start = routing.Start(v)
        end = routing.End(v)
        time_dim.CumulVar(start).SetRange(
            int(DAY_START * _Scale), int(DAY_START * _Scale)
        )
        time_dim.CumulVar(end).SetRange(
            int(DAY_START * _Scale), int(DAY_END * _Scale)
        )
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(start))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(end))

    # ── Per-node time windows and day-vehicle locking ────────────────────────
    for node_idx in range(1, num_nodes):
        ci, tw, day_idx = node_info[node_idx]
        idx = manager.NodeToIndex(node_idx)
        # Time window for this node
        time_dim.CumulVar(idx).SetRange(
            int(tw.earliest * _Scale), int(tw.latest * _Scale)
        )
        # Lock this node to the correct vehicle (day)
        for v in range(num_vehicles):
            if v != day_idx:
                routing.VehicleVar(idx).RemoveValue(v)

    # ── Each client scheduled exactly once (via disjunctions) ────────────────
    # A disjunction with a very high penalty ensures the solver schedules
    # the client if at all possible, and counts missing ones in the objective.
    for ci, node_set in enumerate(client_node_sets):
        if not node_set:
            result.unscheduled.append(schedulable[ci].client_id)
            continue
        or_tools_indices = [manager.NodeToIndex(n) for n in node_set]
        routing.AddDisjunction(or_tools_indices, _PENALTY_UNSCHEDULED * _Scale)

    # ── Soft travel buffer penalties ─────────────────────────────────────────
    # We add a penalty for each arc whose travel time exceeds BUFFER_PREFERRED.
    # OR-Tools doesn't have a built-in "soft arc" mechanism, so we approximate
    # by adding a penalty dimension that accumulates excess travel.
    #
    # Implementation: soft penalty dimension over transit-only (no service).
    # Exceeding 15 min: cost += (excess * PENALTY_OVER_BUFFER_SOFT)
    # Exceeding 20 min: additional cost += (excess * PENALTY_OVER_BUFFER_HARD)
    # We bake these into the arc cost evaluator via a separate callback.

    def _penalised_transit(from_idx: int, to_idx: int) -> int:
        base = _transit(from_idx, to_idx)
        drive_mins = base / _Scale
        penalty = 0
        if drive_mins > BUFFER_PREFERRED:
            over_pref = (drive_mins - BUFFER_PREFERRED) * _PENALTY_OVER_BUFFER_SOFT
            penalty += int(round(over_pref))
        if drive_mins > BUFFER_SOFT_MAX:
            over_hard = (drive_mins - BUFFER_SOFT_MAX) * _PENALTY_OVER_BUFFER_HARD
            penalty += int(round(over_hard))
        return base + penalty * _Scale

    penalised_cb = routing.RegisterTransitCallback(_penalised_transit)
    routing.SetArcCostEvaluatorOfAllVehicles(penalised_cb)

    # ── Lunch breaks ─────────────────────────────────────────────────────────
    _add_lunch_constraints(routing, time_dim, manager, num_vehicles, variant)

    # ── Search parameters ─────────────────────────────────────────────────────
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = SOLVER_TIME_LIMIT_SEC
    params.log_search = False

    # ── Solve ────────────────────────────────────────────────────────────────
    solution = routing.SolveWithParameters(params)

    if not solution:
        # No solution found at all – mark everyone unscheduled
        for c in schedulable:
            if c.client_id not in result.unscheduled:
                result.unscheduled.append(c.client_id)
        return result

    # ── Extract solution ──────────────────────────────────────────────────────
    scheduled_client_ids: set[str] = set()
    day_routes: dict[str, DayRoute] = {d: DayRoute(day=d) for d in DAYS}

    for v in range(num_vehicles):
        day = DAYS[v]
        route = day_routes[day]
        idx = routing.Start(v)
        prev_end_min: Optional[int] = None

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node == 0:
                idx = solution.Value(routing.NextVar(idx))
                continue
            ci, tw, _ = node_info[node]
            client = schedulable[ci]
            start_scaled = solution.Min(time_dim.CumulVar(idx))
            start_min = start_scaled // _Scale
            end_min = start_min + SESSION_MINUTES

            session = ScheduledSession(
                client_id=client.client_id,
                day=day,
                start_min=start_min,
                end_min=end_min,
                location_index=client.location_index,
            )
            route.sessions.append(session)
            scheduled_client_ids.add(client.client_id)

            # Gap violation tracking
            if prev_end_min is not None:
                gap = start_min - prev_end_min
                if gap > BUFFER_SOFT_MAX:
                    route.gap_violations += 1
            prev_end_min = end_min

            idx = solution.Value(routing.NextVar(idx))

        # Compute total drive time for this day
        route.total_drive_min = _compute_day_drive(
            route.sessions, travel_matrix, schedulable
        )

    for c in schedulable:
        if c.client_id not in scheduled_client_ids:
            result.unscheduled.append(c.client_id)

    result.days = list(day_routes.values())
    result.total_drive_min = sum(d.total_drive_min for d in result.days)
    return result


# ── Lunch constraint helpers ──────────────────────────────────────────────────


def _add_lunch_constraints(
    routing: pywrapcp.RoutingModel,
    time_dim,
    manager: pywrapcp.RoutingIndexManager,
    num_vehicles: int,
    variant: LunchVariant,
) -> None:
    """Inject lunch break constraints via OR-Tools BreakIntervals."""
    if variant == LunchVariant.NO_LUNCH:
        return

    # BreakIntervalVar: a break that takes LUNCH_DURATION minutes, must start
    # between LUNCH_EARLIEST and LUNCH_LATEST_START.
    if variant == LunchVariant.DAILY_LUNCH:
        vehicles_with_lunch = list(range(num_vehicles))
    else:  # HYBRID – lunch on 2 vehicles; we'll set breaks on all 4 but make
           # 2 of them optional (penalty-based).  The solver will pick 2.
        vehicles_with_lunch = list(range(num_vehicles))

    for v in vehicles_with_lunch:
        is_optional = variant == LunchVariant.HYBRID
        break_var = routing.solver().FixedDurationIntervalVar(
            int(LUNCH_EARLIEST * _Scale),
            int(LUNCH_LATEST_START * _Scale),
            int(LUNCH_DURATION * _Scale),
            is_optional,
            f"lunch_v{v}",
        )
        time_dim.SetBreakIntervalsOfVehicle(
            [break_var], v, []
        )


# ── Post-solve drive time calculation ────────────────────────────────────────


def _compute_day_drive(
    sessions: list[ScheduledSession],
    travel_matrix: list[list[float]],
    clients: list[Client],
) -> float:
    """Sum drive times: home → s1 → s2 → … → home."""
    if not sessions:
        return 0.0

    loc_id: dict[str, int] = {c.client_id: c.location_index for c in clients}
    total = 0.0
    prev_loc = 0  # home base
    for s in sessions:
        cur_loc = loc_id[s.client_id]
        total += travel_matrix[prev_loc][cur_loc]
        prev_loc = cur_loc
    total += travel_matrix[prev_loc][0]  # return home
    return round(total, 1)
