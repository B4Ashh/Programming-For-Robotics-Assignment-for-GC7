# =============================================================================
# vacuum_controller.py — Main Controller & System Integration
# =============================================================================
# ACT layer of the Sense-Think-Act architecture.
# Entry point for the Pioneer 3-DX Webots controller.
#
# System Architecture (Sense-Think-Act):
#   SENSE:  sensors.py     -> sonar zones, person detection,
#                             collision severity, predictive check
#   THINK:  navigation.py  -> FSM, lawnmower planner, recovery
#   ACT:    this file      -> wheel velocity commands
#   MAP:    grid_map.py    -> occupancy grid, coverage tracking
#
# SAFETY AND FAIL-SAFE MECHANISMS:
#   1.  Battery low (20%)           -> RETURN_HOME state
#   2.  Sweep complete              -> RETURN_HOME state
#   3.  Predictive collision        -> intercepts before sonar threshold
#   4.  Collision severity          -> graduated 3-phase recovery
#   5.  Dual stall detector         -> position AND coverage based
#   6.  Person detection            -> PAUSED with 5-tick timeout
#   7.  Sonar MIN/MAX filter        -> rejects bad sensor readings
#   8.  side_min zone               -> catches blind-spot obstacles
#   9.  Speed ramp near home        -> prevents overshoot
#   10. Hard motor stop on DONE     -> prevents runaway
#   11. Return home stuck detector  -> persists across escapes, tries
#                                      alternating turn directions
#   12. Max escape attempts (8)     -> stops gracefully if permanently blocked
#   13. GPS-based obstacle marking  -> uses actual sonar distance, never
#                                      overwrites arena wall cells
# =============================================================================

import math
from controller import Robot
from grid_map   import (GridMap, ARENA_MAX_X, ARENA_MIN_X)
from sensors    import sonar_zones, obstacle_hits, person_nearby
from navigation import Navigator

TIME_STEP        = 64       # simulation timestep in milliseconds
MAX_STEPS        = 45000    # total steps = total battery life
MAX_SPEED        = 6.0      # maximum wheel speed (rad/s)
FORCE_TURN_STEPS = 80       # sweep escape duration in steps
STALL_LIMIT      = 3        # stall ticks before triggering escape

robot    = Robot()
timestep = int(robot.getBasicTimeStep())

# ── Motors ────────────────────────────────────────────────────────────
left_motor  = robot.getDevice('left wheel')
right_motor = robot.getDevice('right wheel')
left_motor.setPosition(float('inf'))   # velocity control mode
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

print("Left motor:",  left_motor)
print("Right motor:", right_motor)

# ── Sonars (so0-so7 front arc, so8-so15 rear arc) ─────────────────────
sonars = []
for i in range(16):
    s = robot.getDevice(f'so{i}')
    s.enable(timestep)
    sonars.append(s)

# ── GPS + Compass ─────────────────────────────────────────────────────
gps = robot.getDevice('gps')
gps.enable(timestep)
compass = robot.getDevice('compass')
compass.enable(timestep)

# ── Warmup: let sensors stabilise before reading ──────────────────────
for _ in range(20):
    robot.step(timestep)

# ── Initial pose ──────────────────────────────────────────────────────
pos            = gps.getValues()
init_x, init_y = pos[0], pos[1]
print(f"Robot start: ({init_x:.2f}, {init_y:.2f})")

# ── Setup ─────────────────────────────────────────────────────────────
gmap       = GridMap()
nav        = Navigator(gmap, init_x, init_y)
step_count = 0
last_lv    = 0.0    # left velocity from previous step (for predict_collision)
last_rv    = 0.0    # right velocity from previous step

# ── Robot trail — (grid_row, grid_col, heading_rad) every 50 steps ────
# Stores heading so arrows can show direction of travel on grid map
robot_trail = []

# ── Dual stall detection ──────────────────────────────────────────────
# Fires when EITHER position hasn't changed (stuck against wall/box)
# OR coverage hasn't changed (revisiting already-cleaned area).
# Checked every 100 steps (~6.4 seconds).
stall_ticks        = 0
stall_last_x       = init_x
stall_last_y       = init_y
last_coverage      = 0.0
force_turn_counter = 0    # counts down escape manoeuvre steps
force_turn_dir     = 1    # alternates +1/-1 each escape to try both sides

# ROW_SKIP: number of waypoints to jump when a stall is detected.
# Roughly half a row — enough to escape the stuck area entirely.
ROW_SKIP = max(1, int((ARENA_MAX_X - ARENA_MIN_X - 0.6) / 0.25 / 2))

# ── Return home stuck detection ───────────────────────────────────────
# Separate from sweep stall detection — used only during RETURN_HOME.
# Attempt counter PERSISTS across individual escape cycles and only
# resets when the robot makes genuine 0.5m progress toward home.
# This prevents the robot from being stuck on attempt 1 forever.
# Odd attempts turn left, even attempts turn right — systematic search.
home_stuck_ticks      = 0
HOME_STUCK_LIMIT      = 3
home_last_x           = init_x
home_last_y           = init_y
home_escape_counter   = 0
HOME_ESCAPE_STEPS     = 150   # longer than sweep escape — walls need space
home_escape_attempts  = 0
HOME_ESCAPE_MAX       = 8     # give up and stop if still stuck after 8 tries
home_progress_x       = init_x   # tracks genuine progress toward home
home_progress_y       = init_y


# ── HTML grid visualisation with directional trail arrows ─────────────
def export_grid_html(gmap, step, coverage, rx, ry, trail):
    """
    Export the occupancy grid as an auto-refreshing HTML page.
    Open grid_map.html in a browser during simulation to watch live.

    Colours:
      Green  = floor that has been cleaned (VISITED)
      Red    = confirmed obstacle (GPS-marked on AVOID entry)
      Dark   = arena wall boundary (WALL)
      Grey   = floor not yet cleaned (UNVISITED)
      Gold   = robot current position
      Blue   = directional trail arrows showing path taken
    """
    from grid_map import VISITED, OBSTACLE, WALL, UNVISITED

    COLORS = {
        UNVISITED: '#e8e8e8',
        VISITED:   '#4CAF50',
        OBSTACLE:  '#e74c3c',
        WALL:      '#2c3e50',
    }

    rows, cols = gmap.grid.shape
    cell_px    = 12

    rrow, rcol = gmap.world_to_grid(rx, ry)
    draw_rrow  = rows - 1 - rrow
    robot_px_x = rcol      * cell_px + cell_px // 2
    robot_px_y = draw_rrow * cell_px + cell_px // 2

    # Home position — fixed from start, shown as a gold star
    hrow, hcol  = gmap.world_to_grid(nav.start_x, nav.start_y)
    draw_hrow   = rows - 1 - hrow
    home_px_x   = hcol      * cell_px + cell_px // 2
    home_px_y   = draw_hrow * cell_px + cell_px // 2

    lines = [
        '<!DOCTYPE html><html><head>',
        '<meta http-equiv="refresh" content="2">',
        '<title>Vacuum Robot Grid</title>',
        '<style>',
        'body{font-family:monospace;background:#111;'
        'color:#eee;padding:20px}',
        'h2{margin-bottom:6px}',
        'p{color:#aaa;font-size:13px;margin:4px 0}',
        '.legend{display:flex;gap:20px;margin:10px 0;flex-wrap:wrap;'
        'align-items:center}',
        '.swatch{display:inline-block;width:14px;height:14px;'
        'vertical-align:middle;margin-right:4px;border:1px solid #555}',
        '.grid-wrap{position:relative;display:inline-block;'
        'margin-top:8px}',
        'table{border-collapse:collapse;border:2px solid #333}',
        f'td{{width:{cell_px}px;height:{cell_px}px;padding:0;'
        f'border:1px solid #33333355}}',
        '.robot{position:absolute;width:10px;height:10px;'
        'background:gold;border-radius:50%;border:2px solid #fff;'
        'transform:translate(-50%,-50%);pointer-events:none;z-index:10}',
        '.home{position:absolute;pointer-events:none;z-index:9;'
        'transform:translate(-50%,-50%)}',
        '.arrow{position:absolute;pointer-events:none;z-index:5;'
        'transform-origin:center center;}',
        '</style></head><body>',
        f'<h2>Vacuum Robot Grid &mdash; Step {step} '
        f'&mdash; Coverage {coverage:.1f}%</h2>',
        '<div class="legend">',
        '<span><span class="swatch" '
        'style="background:#2c3e50"></span>Wall</span>',
        '<span><span class="swatch" '
        'style="background:#e74c3c"></span>Obstacle (GPS)</span>',
        '<span><span class="swatch" '
        'style="background:#4CAF50"></span>Cleaned</span>',
        '<span><span class="swatch" '
        'style="background:#e8e8e8"></span>Unvisited</span>',
        '<span><span class="swatch" '
        'style="background:gold;border-radius:50%;border-color:gold">'
        '</span>Robot</span>',
        '<span><svg width="14" height="14" viewBox="0 0 14 14">'
        '<polygon points="7,1 13,13 7,10 1,13" '
        'fill="#4488ff" opacity="0.85"/></svg>'
        '&nbsp;Trail</span>',
        '<span><svg width="14" height="14" viewBox="0 0 14 14">'
        '<polygon points="7,0 8.5,5 14,5 9.5,8 11,14 7,10.5 3,14 4.5,8 0,5 5.5,5" '
        'fill="#ff4dff" stroke="#fff" stroke-width="0.5"/></svg>'
        '&nbsp;Home</span>',
        '</div>',
        '<div class="grid-wrap"><table>',
    ]

    for r in range(rows - 1, -1, -1):
        lines.append('<tr>')
        for c in range(cols):
            color = COLORS.get(int(gmap.grid[r][c]), '#ffffff')
            lines.append(f'<td style="background:{color}"></td>')
        lines.append('</tr>')

    lines.append(
        f'<div class="robot" style="left:{robot_px_x}px;'
        f'top:{robot_px_y}px"></div>')

    # Home star — fixed position, visible from the very first export
    lines.append(
        f'<svg class="home" width="16" height="16" viewBox="0 0 16 16" '
        f'style="left:{home_px_x}px;top:{home_px_y}px">'
        f'<polygon points="8,0 9.5,5.5 16,5.5 10.5,9 12.5,16 8,12 3.5,16 5.5,9 0,5.5 6.5,5.5" '
        f'fill="#ff4dff" stroke="#fff" stroke-width="0.8"/>'
        f'</svg>')

    # Trail arrows — each rotated to match actual heading at that point
    # screen_deg converts world heading to CSS rotation:
    #   heading 0 (East) = arrow points right = 90 deg rotation
    #   heading pi/2 (North/up on screen) = 0 deg rotation
    for (tr, tc, heading) in trail:
        draw_tr    = rows - 1 - tr
        ax         = tc      * cell_px + cell_px // 2
        ay         = draw_tr * cell_px + cell_px // 2
        screen_deg = -math.degrees(heading) + 90
        lines.append(
            f'<svg class="arrow" width="10" height="10" '
            f'viewBox="0 0 10 10" '
            f'style="left:{ax - 5}px;top:{ay - 5}px;'
            f'transform:rotate({screen_deg:.0f}deg)">'
            f'<polygon points="5,0 9,10 5,7 1,10" '
            f'fill="#4488ff" opacity="0.75"/>'
            f'</svg>')

    lines += [
        '</div>',
        f'<p>cell=20cm &nbsp;|&nbsp; Arena=10mx10m &nbsp;|&nbsp; '
        f'State: {nav.state} &nbsp;|&nbsp; '
        f'WP: {nav.wp_index}/{len(nav.waypoints)} &nbsp;|&nbsp; '
        f'Trail: {len(trail)} arrows &nbsp;|&nbsp; '
        f'Auto-refreshes every 2s</p>',
        '</body></html>',
    ]

    with open('grid_map.html', 'w') as f:
        f.write('\n'.join(lines))


# ── Main simulation loop ──────────────────────────────────────────────
while robot.step(timestep) != -1:
    step_count  += 1
    # Battery fraction: starts at 1.0, reaches 0.0 at MAX_STEPS
    battery_frac = max(0.0, 1.0 - (step_count / MAX_STEPS))

    # ── SENSE: read all sensors ───────────────────────────────────────
    pos    = gps.getValues()
    rx, ry = pos[0], pos[1]         # world position in metres
    north  = compass.getValues()
    rtheta = math.atan2(north[0], north[2])  # heading in radians

    readings = [s.getValue() for s in sonars]
    zones    = sonar_zones(readings)
    person   = person_nearby(readings)

    # ── MAP: mark floor under robot as visited ────────────────────────
    # 3x3 footprint accounts for the robot's physical width (~40cm)
    row, col = gmap.world_to_grid(rx, ry)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            gmap.mark_visited(row + dr, col + dc)

    # Record trail point every 50 steps (row, col, heading)
    if step_count % 50 == 0:
        robot_trail.append((row, col, rtheta))
        if len(robot_trail) > 500:
            robot_trail.pop(0)

    # ── SAFETY: dual stall detection every 100 steps ─────────────────
    # Checked every 100 steps (~6.4 seconds at 64ms per step)
    if step_count % 100 == 0:
        coverage = gmap.coverage_percent()

        if nav.state == 'SWEEP':
            moved         = math.hypot(rx - stall_last_x,
                                       ry - stall_last_y)
            coverage_gain = abs(coverage - last_coverage)

            # Dual condition: stuck if either position OR coverage frozen
            if moved < 0.25 or coverage_gain < 0.05:
                stall_ticks += 1
                print(f"  [stall] tick {stall_ticks}/{STALL_LIMIT} "
                      f"moved={moved:.2f}m "
                      f"coverage_gain={coverage_gain:.2f}% "
                      f"coverage={coverage:.1f}%")
            else:
                stall_ticks = 0

            if stall_ticks >= STALL_LIMIT:
                print(f"  [stall] stuck — "
                      f"skipping {ROW_SKIP} waypoints "
                      f"and driving perpendicular")
                # Jump forward in waypoint list past the stuck area
                nav.wp_index = min(
                    nav.wp_index + ROW_SKIP,
                    len(nav.waypoints) - 1)
                force_turn_counter  = FORCE_TURN_STEPS * 2
                force_turn_dir     *= -1   # alternate direction each time
                stall_ticks         = 0

            stall_last_x  = rx
            stall_last_y  = ry
            last_coverage = coverage
        else:
            # Reset stall tracking when not sweeping
            stall_ticks   = 0
            stall_last_x  = rx
            stall_last_y  = ry
            last_coverage = coverage

        # ── SAFETY: return home stuck detection ───────────────────────
        # Separate stall detection for RETURN_HOME state.
        # Attempt counter only resets on genuine 0.5m progress.
        if nav.state == 'RETURN_HOME':
            moved = math.hypot(rx - home_last_x, ry - home_last_y)
            if moved < 0.20:
                home_stuck_ticks += 1
                print(f"  [home] stuck tick {home_stuck_ticks}/"
                      f"{HOME_STUCK_LIMIT} "
                      f"(moved only {moved:.2f}m)")
                if home_stuck_ticks >= HOME_STUCK_LIMIT:
                    home_escape_attempts += 1
                    if home_escape_attempts > HOME_ESCAPE_MAX:
                        print(f"  [home] max escape attempts reached "
                              f"— stopping in place")
                        nav.state = 'DONE'
                    else:
                        print(f"  [home] escape attempt "
                              f"{home_escape_attempts}/{HOME_ESCAPE_MAX}"
                              f" — backing up and rerouting")
                        home_escape_counter = HOME_ESCAPE_STEPS
                    home_stuck_ticks = 0
            else:
                home_stuck_ticks = 0
                # Only reset attempt counter on genuine progress toward home
                real_progress = math.hypot(
                    rx - home_progress_x, ry - home_progress_y)
                if real_progress > 0.5:
                    home_escape_attempts = 0
                    home_progress_x = rx
                    home_progress_y = ry
            home_last_x = rx
            home_last_y = ry

        print(f"[{step_count}] state={nav.state}  "
              f"coverage={coverage:.1f}%  "
              f"battery={battery_frac*100:.0f}%  "
              f"wp={nav.wp_index}/{len(nav.waypoints)}")

    # Export HTML grid every 300 steps
    if step_count % 300 == 0:
        export_grid_html(gmap, step_count,
                         gmap.coverage_percent(), rx, ry,
                         robot_trail)

    # ── ACT: escape sequences take priority over FSM ──────────────────
    # The 'continue' statements skip the FSM entirely during escapes.

    # Sweep escape: back up, turn perpendicular, drive forward
    if force_turn_counter > 0 and nav.state != 'RETURN_HOME':
        force_turn_counter -= 1
        total = FORCE_TURN_STEPS * 2
        if force_turn_counter > total * 0.6:
            lv, rv = -MAX_SPEED * 0.35, -MAX_SPEED * 0.35
            print(f"  [escape] backing up "
                  f"({force_turn_counter} steps left)") \
                if force_turn_counter % 20 == 0 else None
        elif force_turn_counter > total * 0.3:
            lv =  1.5 * force_turn_dir
            rv = -1.5 * force_turn_dir
            print(f"  [escape] turning perpendicular "
                  f"({force_turn_counter} steps left)") \
                if force_turn_counter % 20 == 0 else None
        else:
            lv, rv = MAX_SPEED, MAX_SPEED
            print(f"  [escape] driving forward "
                  f"({force_turn_counter} steps left)") \
                if force_turn_counter % 20 == 0 else None
        left_motor.setVelocity(lv)
        right_motor.setVelocity(rv)
        last_lv, last_rv = lv, rv
        continue

    # Return home escape: back up, turn toward home, drive forward clear
    # Odd attempts turn left, even attempts turn right — tries both sides
    if home_escape_counter > 0 and nav.state == 'RETURN_HOME':
        home_escape_counter -= 1
        total = HOME_ESCAPE_STEPS
        if home_escape_counter > total * 0.55:
            # Phase 1: back up further to clear wall
            lv, rv = -MAX_SPEED * 0.5, -MAX_SPEED * 0.5
            print(f"  [home escape] attempt {home_escape_attempts} "
                  f"backing up ({home_escape_counter} steps left)") \
                if home_escape_counter % 20 == 0 else None
        elif home_escape_counter > total * 0.20:
            # Phase 2: alternate turn direction each attempt
            if home_escape_attempts % 2 == 1:
                lv, rv = -2.5, 2.5    # odd attempt: turn left
            else:
                lv, rv = 2.5, -2.5    # even attempt: turn right
            home_err = nav._heading_to_home(rx, ry, rtheta)
            print(f"  [home escape] attempt {home_escape_attempts} "
                  f"turning "
                  f"{'left' if home_escape_attempts % 2 == 1 else 'right'} "
                  f"(err={math.degrees(home_err):.0f}deg "
                  f"{home_escape_counter} steps left)") \
                if home_escape_counter % 20 == 0 else None
        else:
            # Phase 3: drive forward to clear the wall
            lv, rv = MAX_SPEED * 0.6, MAX_SPEED * 0.6
            print(f"  [home escape] attempt {home_escape_attempts} "
                  f"driving clear ({home_escape_counter} steps left)") \
                if home_escape_counter % 20 == 0 else None
        left_motor.setVelocity(lv)
        right_motor.setVelocity(rv)
        last_lv, last_rv = lv, rv
        continue

    # ── ACT: normal FSM navigation ────────────────────────────────────
    lv, rv = nav.step(rx, ry, rtheta, zones, person,
                      battery_frac, last_lv, last_rv)
    left_motor.setVelocity(lv)
    right_motor.setVelocity(rv)
    last_lv, last_rv = lv, rv

    # SAFETY: hard motor stop when task complete
    if nav.state == 'DONE':
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        export_grid_html(gmap, step_count,
                         gmap.coverage_percent(), rx, ry,
                         robot_trail)
        print(f"  [nav] DONE — "
              f"final coverage={gmap.coverage_percent():.1f}%")
        break