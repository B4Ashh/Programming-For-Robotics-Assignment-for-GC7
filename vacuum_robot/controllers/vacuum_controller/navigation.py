# =============================================================================
# navigation.py — Navigation & Environment Module
# =============================================================================
# THINK layer of the Sense-Think-Act architecture.
#
# FSM States:
#   STARTUP     -> Drive away from corner into open space before sweeping
#   SWEEP       -> Execute lawnmower coverage pattern waypoint by waypoint
#   AVOID       -> Three-phase collision recovery (back up, turn, forward)
#   PAUSED      -> Person detected — stop and wait until clear
#   RETURN_HOME -> Low battery OR sweep complete — navigate back to start
#   DONE        -> Home reached or max escapes exceeded — hard stop
#
# CONTEXT-AWARE REAL-TIME DECISION PRIORITY (evaluated every 64ms):
#   1. Person nearby    -> PAUSED      (perception safety — highest priority)
#   2. Battery low      -> RETURN_HOME (fail-safe)
#   3. Sweep complete   -> RETURN_HOME (task complete — go home)
#   4. Predicted hit    -> AVOID       (predictive safety)
#   5. Obstacle close   -> AVOID       (reactive safety)
#   6. Normal           -> SWEEP       (task execution — lowest priority)
#
# SAFETY MECHANISMS:
#   1. Battery threshold (20%) triggers RETURN_HOME
#   2. Sweep completion also triggers RETURN_HOME (robot always goes home)
#   3. Predictive collision intercepts before robot physically hits
#   4. Three-phase recovery: backup->turn->forward
#   5. Person pause with 5-tick timeout (won't wait forever)
#   6. GPS-based obstacle marking using actual measured distance
#   7. Speed ramp-down near home prevents overshoot
#   8. Wall-aware steering blends heading with reactive avoidance
# =============================================================================

import math
from grid_map import (ARENA_MIN_X, ARENA_MAX_X,
                      ARENA_MIN_Y, ARENA_MAX_Y,
                      OBSTACLE, WALL)
from sensors  import collision_severity, predict_collision

MAX_SPEED          = 6.0    # maximum wheel velocity (rad/s)
ARRIVE_DIST        = 0.45   # distance at which a waypoint counts as reached
HOME_SLOW_DIST     = 1.5    # start slowing down when this close to home
STRIP_SPACING      = 0.25   # distance between lawnmower rows (metres)
WP_SPACING         = 0.25   # distance between waypoints within a row
BATTERY_LOW        = 0.20   # go home when battery fraction drops below this
PERSON_PAUSE_LIMIT = 5      # maximum ticks to wait for person to move

STARTUP     = 'STARTUP'
SWEEP       = 'SWEEP'
AVOID       = 'AVOID'
PAUSED      = 'PAUSED'
RETURN_HOME = 'RETURN_HOME'
DONE        = 'DONE'


def _angle_error(target, current):
    """
    Shortest-path signed angle error, normalised to [-pi, pi].
    Positive = need to turn left, negative = need to turn right.
    The modulo arithmetic handles wrap-around (e.g. from 170 deg to -170 deg
    is only 20 degrees, not 340 degrees the long way round).
    """
    e = target - current
    return (e + math.pi) % (2 * math.pi) - math.pi


def _frange(start, stop, step):
    """Float range generator supporting positive and negative steps."""
    vals, x = [], start
    if step > 0:
        while x <= stop + 1e-9:
            vals.append(round(x, 3))
            x += step
    else:
        while x >= stop - 1e-9:
            vals.append(round(x, 3))
            x += step
    return vals


class Navigator:
    def __init__(self, grid_map, start_x, start_y):
        self.grid           = grid_map
        self.start_x        = start_x
        self.start_y        = start_y
        self.state          = STARTUP
        self.avoid_timer    = 0
        self.recovery_steps = 0
        self.recovery_phase = 0
        self.stuck_count    = 0
        self.person_ticks   = 0
        self.wp_index       = 0    # index of next waypoint to drive to
        self.waypoints      = []   # full ordered list of (x,y) positions

        # Startup target: drive 1.5m forward from start into open space
        self.startup_tx = start_x
        self.startup_ty = min(start_y + 1.5, ARENA_MAX_Y - 0.8)
        print(f"  [nav] startup -> ({self.startup_tx:.2f},"
              f"{self.startup_ty:.2f})")

    def _build_lawnmower(self, from_x, from_y):
        """
        Generate the full list of waypoints covering the arena floor.

        Uses boustrophedon (lawnmower / back-and-forth) pattern:
        - Rows run left-to-right, then right-to-left, alternating
        - MARGIN = 0.3m keeps waypoints 30cm inside the arena walls
        - Starts from the row nearest the robot's current position,
          sweeps up to the top, then wraps back down to cover lower rows

        The upward-then-downward ordering ensures both the top and bottom
        halves of the arena are covered — if rows were sorted by proximity
        the robot would clean the middle, run out of battery, and leave
        both ends untouched.
        """
        MARGIN = 0.3
        x_min  = ARENA_MIN_X + MARGIN
        x_max  = ARENA_MAX_X - MARGIN
        y_min  = ARENA_MIN_Y + MARGIN
        y_max  = ARENA_MAX_Y - MARGIN

        # Build list of all Y row positions from bottom to top
        y_vals = []
        y = y_min
        while y <= y_max + 1e-9:
            y_vals.append(round(y, 3))
            y += STRIP_SPACING

        # Find nearest row to robot's current position
        nearest_idx = min(range(len(y_vals)),
                          key=lambda i: abs(y_vals[i] - from_y))

        # Order: current row upward to top, then wrap down to bottom
        above   = y_vals[nearest_idx:]                  # current -> top
        below   = list(reversed(y_vals[:nearest_idx]))  # current -> bottom
        ordered = above + below

        waypoints = []
        for i, y in enumerate(ordered):
            if i % 2 == 0:
                # Even rows: left to right
                row_pts = [(x, y) for x in _frange(x_min, x_max,
                                                    WP_SPACING)]
            else:
                # Odd rows: right to left (boustrophedon reversal)
                row_pts = [(x, y) for x in _frange(x_max, x_min,
                                                    -WP_SPACING)]
            if i == 0:
                # First row: start from point nearest the robot
                row_pts.sort(key=lambda p: math.hypot(
                    p[0] - from_x, p[1] - from_y))
            waypoints.extend(row_pts)

        print(f"  [nav] {len(waypoints)} waypoints built "
              f"({len(ordered)} rows, "
              f"y={ordered[0]:.2f} -> {y_max:.2f} -> {ordered[-1]:.2f})")
        return waypoints

    def _steer(self, rx, ry, rtheta, tx, ty, gain=3.5):
        """
        Proportional heading controller — steers toward a target (tx, ty).

        Computes the desired heading angle to the target, finds the
        shortest angular error from the current heading, and converts
        that into a left/right wheel speed difference (differential
        steering). Positive error -> left wheel slower -> turns left.
        """
        desired = math.atan2(ty - ry, tx - rx)
        err     = _angle_error(desired, rtheta)
        turn    = err * gain
        lv = max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED - turn))
        rv = max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED + turn))
        return lv, rv

    def _wall_avoidance_steer(self, rx, ry, rtheta, tx, ty,
                               zones, speed):
        """
        Steering toward a target while reactively avoiding walls.

        Used during RETURN_HOME so the robot doesn't drive straight
        into a wall that happens to sit between it and the home position.
        Blends proportional heading toward target with reactive pushes:
          - If something ahead: turn toward whichever side is more open
          - If too close on left: push right (turn correction added)
          - If too close on right: push left
        """
        desired = math.atan2(ty - ry, tx - rx)
        err     = _angle_error(desired, rtheta)
        turn    = err * 3.0

        f = zones['front_wide']
        l = zones['left']
        r = zones['right']

        if f < 0.35:
            wall_turn = 3.0 if l > r else -3.0
            turn += wall_turn
        elif l < 0.30:
            turn -= 2.0   # push away from left wall
        elif r < 0.30:
            turn += 2.0   # push away from right wall

        lv = max(-speed, min(speed, speed - turn))
        rv = max(-speed, min(speed, speed + turn))
        return lv, rv

    def _heading_to_home(self, rx, ry, rtheta):
        """
        Return the signed angle error from current heading to home.
        Used by the home escape logic to turn in the correct direction
        after backing away from a wall.
        Positive = need to turn left to face home.
        Negative = need to turn right to face home.
        """
        desired = math.atan2(self.start_y - ry, self.start_x - rx)
        return _angle_error(desired, rtheta)

    def _avoid(self, zones):
        """
        Three-phase obstacle recovery manoeuvre.

        Phase 0 — Reverse: back away to create clearance.
                  If rear is clear, reverse straight back.
                  If rear is also blocked, pivot while reversing.
        Phase 1 — Turn: rotate toward whichever side has more open space.
                  Left clearance > right clearance -> turn left, else right.
        Phase 2 — Forward: drive into the cleared space.
                  If front clears, drive forward.
                  If still blocked, restart from phase 0.

        recovery_steps counts down the remaining steps in each phase.
        recovery_phase tracks which phase is currently active.
        """
        f    = zones['front_wide']
        l    = zones['left']
        r    = zones['right']
        rear = zones['rear']

        if self.recovery_steps > 0:
            self.recovery_steps -= 1

            if self.recovery_phase == 0:   # Reversing
                if rear > 0.25:
                    return -MAX_SPEED * 0.5, -MAX_SPEED * 0.5
                else:
                    self.recovery_phase = 1
                    self.recovery_steps = 30
                    return -MAX_SPEED * 0.4, MAX_SPEED * 0.4

            elif self.recovery_phase == 1:  # Turning
                if l > r:
                    return -MAX_SPEED * 0.5,  MAX_SPEED * 0.5
                else:
                    return  MAX_SPEED * 0.5, -MAX_SPEED * 0.5

            elif self.recovery_phase == 2:  # Driving forward
                if f > 0.30:
                    return MAX_SPEED * 0.6, MAX_SPEED * 0.6
                else:
                    self.recovery_phase = 0
                    self.recovery_steps = 15
                    return -MAX_SPEED * 0.5, -MAX_SPEED * 0.5

            return 0.0, 0.0

        # Recovery complete — classify current situation for response
        severity = collision_severity(zones)

        if severity == 'stuck':
            self.stuck_count    += 1
            self.recovery_phase  = 0
            self.recovery_steps  = 25
            print(f"  [avoid] stuck! recovery #{self.stuck_count}")
            return -MAX_SPEED * 0.5, -MAX_SPEED * 0.5

        elif severity == 'danger':
            self.recovery_phase = 0
            self.recovery_steps = 15
            return -MAX_SPEED * 0.5, -MAX_SPEED * 0.5

        elif severity == 'warning':
            if f < 0.30:
                return (-1.2, 1.8) if l > r else (1.8, -1.2)
            if l < 0.22:
                return MAX_SPEED * 0.8, MAX_SPEED * 0.2
            if r < 0.22:
                return MAX_SPEED * 0.2, MAX_SPEED * 0.8

        return MAX_SPEED * 0.4, MAX_SPEED * 0.4

    def step(self, rx, ry, rtheta, zones, person_near,
             battery_frac, last_lv=0.0, last_rv=0.0):
        """
        Main FSM step — called every simulation timestep (64ms).

        Two parts:
        1. TRANSITION LOGIC — decide whether to change state
        2. VELOCITY COMMANDS — given the current state, what speeds?

        Returns (left_wheel_speed, right_wheel_speed).
        """

        # ── STARTUP: drive into open space before beginning sweep ─────
        if self.state == STARTUP:
            dist = math.hypot(rx - self.startup_tx, ry - self.startup_ty)
            if dist < ARRIVE_DIST:
                # Reached startup position — build waypoints and begin sweep
                self.waypoints = self._build_lawnmower(rx, ry)
                self.wp_index  = 0
                self.state     = SWEEP
                print(f"  [nav] starting sweep from ({rx:.2f},{ry:.2f})")
                return 0.0, 0.0
            if zones['front_wide'] < 0.30:
                # Obstacle in the way during startup — dodge it
                return (2.0, -2.0) if zones['left'] < zones['right'] \
                       else (-2.0, 2.0)
            return self._steer(rx, ry, rtheta,
                               self.startup_tx, self.startup_ty)

        # ── FSM TRANSITION LOGIC ──────────────────────────────────────
        if self.state == SWEEP:
            if person_near:
                # Person detected — pause immediately
                self.person_ticks = 0
                self.state        = PAUSED
                print("  [nav] paused — person nearby")

            elif battery_frac < BATTERY_LOW:
                # Battery getting low — return home now
                self.state = RETURN_HOME
                print("  [nav] low battery — going home")

            elif self.wp_index >= len(self.waypoints):
                # All waypoints visited — sweep complete, return home
                self.state = RETURN_HOME
                print("  [nav] sweep complete — returning home")

            else:
                # Check for imminent collision before moving
                will_hit, hit_dir = predict_collision(
                    rx, ry, rtheta, last_lv, last_rv, zones['raw'])
                if will_hit:
                    print(f"  [predict] imminent collision "
                          f"({hit_dir}) — avoiding")
                    # Mark obstacle using actual measured sonar distance
                    self.grid.mark_obstacle_ahead(rx, ry, rtheta, zones)
                    self.state          = AVOID
                    self.avoid_timer    = 0
                    self.recovery_steps = 0
                    self.recovery_phase = 0

                elif (zones['front_wide'] < 0.28
                      or zones['left']     < 0.20
                      or zones['right']    < 0.20
                      or zones['side_min'] < 0.14):
                    # Sonar triggered reactively
                    self.grid.mark_obstacle_ahead(rx, ry, rtheta, zones)
                    self.state          = AVOID
                    self.avoid_timer    = 0
                    self.recovery_steps = 0
                    self.recovery_phase = 0

        elif self.state == AVOID:
            self.avoid_timer += 1

            # Assign initial recovery based on severity at entry
            if self.recovery_steps == 0 and self.avoid_timer == 1:
                severity = collision_severity(zones)
                if severity in ('danger', 'stuck'):
                    self.recovery_phase = 0
                    self.recovery_steps = 15
                elif severity == 'warning':
                    self.recovery_phase = 1
                    self.recovery_steps = 20

            # Progress through recovery phases
            if self.recovery_steps == 0 and self.recovery_phase == 0:
                self.recovery_phase = 1
                self.recovery_steps = 25
            elif self.recovery_steps == 0 and self.recovery_phase == 1:
                self.recovery_phase = 2
                self.recovery_steps = 15

            # Return to SWEEP when all sides are clear
            clear = (zones['front_wide'] > 0.35
                     and zones['left']    > 0.25
                     and zones['right']   > 0.25
                     and zones['side_min'] > 0.16
                     and self.avoid_timer  > 20
                     and self.recovery_steps == 0)
            if clear:
                wp = self.waypoints[self.wp_index] \
                     if self.wp_index < len(self.waypoints) else None
                if wp and math.hypot(rx - wp[0], ry - wp[1]) < 1.0:
                    print(f"  [nav] skipping blocked wp {wp}")
                    self.wp_index += 1
                self.recovery_steps = 0
                self.recovery_phase = 0
                self.avoid_timer    = 0
                self.state          = SWEEP
            if person_near:
                self.person_ticks = 0
                self.state        = PAUSED

        elif self.state == PAUSED:
            self.person_ticks += 1
            if (not person_near
                    or self.person_ticks >= PERSON_PAUSE_LIMIT):
                reason = ("person gone" if not person_near
                          else "pause timeout")
                print(f"  [nav] resuming — {reason} "
                      f"(paused {self.person_ticks} ticks)")
                self.person_ticks = 0
                self.state        = SWEEP

        elif self.state == RETURN_HOME:
            dist = math.hypot(rx - self.start_x, ry - self.start_y)
            if dist < ARRIVE_DIST:
                self.state = DONE
                print("  [nav] home reached — DONE")

        # ── VELOCITY COMMANDS ─────────────────────────────────────────
        if self.state == SWEEP:
            if self.wp_index >= len(self.waypoints):
                self.state = RETURN_HOME
                print("  [nav] sweep complete — returning home")
                return 0.0, 0.0

            wp     = self.waypoints[self.wp_index]
            tx, ty = wp
            dist   = math.hypot(rx - tx, ry - ty)
            if dist < ARRIVE_DIST:
                # Waypoint reached — advance to next one
                self.wp_index += 1
                return MAX_SPEED, MAX_SPEED
            return self._steer(rx, ry, rtheta, tx, ty)

        elif self.state == AVOID:
            return self._avoid(zones)

        elif self.state == PAUSED:
            return 0.0, 0.0

        elif self.state == RETURN_HOME:
            severity = collision_severity(zones)
            if severity == 'stuck':
                return -MAX_SPEED * 0.4, -MAX_SPEED * 0.4
            elif severity == 'danger':
                return self._avoid(zones)

            dist_home = math.hypot(
                rx - self.start_x, ry - self.start_y)

            if dist_home < HOME_SLOW_DIST:
                # Slow down as we approach home to avoid overshoot
                speed = max(1.5, MAX_SPEED * (dist_home / HOME_SLOW_DIST))
                return self._wall_avoidance_steer(
                    rx, ry, rtheta,
                    self.start_x, self.start_y,
                    zones, speed)

            return self._wall_avoidance_steer(
                rx, ry, rtheta,
                self.start_x, self.start_y,
                zones, MAX_SPEED * 0.8)

        elif self.state == DONE:
            return 0.0, 0.0

        return 0.0, 0.0