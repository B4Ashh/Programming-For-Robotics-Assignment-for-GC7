# =============================================================================
# sensors.py — Perception & Sensor Processing Module
# =============================================================================
# SENSE layer of the Sense-Think-Act architecture.
#
# Pioneer 3-DX Sonar Layout (Webots documentation):
#   16 sonars: 8 forward-facing (so0-so7), 8 rear-facing (so8-so15)
#   Angular separation: ~20 degrees between adjacent sensors
#   Wheel radius: 0.0975m | Axle length: 0.381m
#   Max reliable range: 5.0m | Dead zone: <0.02m
#
# SAFETY MECHANISMS:
#   - MIN/MAX filtering rejects uninitialised and out-of-range readings
#   - clean_readings() ensures no zero or inf values reach decision logic
#   - collision_severity() graduated response: none->warning->danger->stuck
#   - predict_collision() checks future position before moving
#
# NOTE ON OBSTACLE MAPPING:
#   Sonar angle offsets on the Pioneer 3-DX are not reliable enough to
#   accurately mark obstacle cells in the grid. Obstacle avoidance is
#   handled reactively by the FSM using zone thresholds. GPS-based marking
#   in grid_map.mark_obstacle_ahead() is used instead on AVOID entry,
#   using the actual measured sonar distance for accurate placement.
# =============================================================================

import math

SONAR_ANGLES_DEG = [
    -90, -50, -30, -10,  10,  30,  50,  90,
     90, 130, 150, 170,-170,-150,-130, -90
]

WHEEL_RADIUS = 0.0975
AXLE_LENGTH  = 0.381

THRESHOLD_DANGER  = 0.20
THRESHOLD_WARNING = 0.35
THRESHOLD_PERSON  = 0.60   # tightened — avoids false triggers from boxes
MAX_SONAR         = 5.0
MIN_SONAR         = 0.02


def clean_readings(readings):
    """
    SAFETY: Replace invalid sonar readings with MAX_SONAR.
    Filters uninitialised zeros, out-of-range values, and infinity.
    Any reading outside MIN_SONAR..MAX_SONAR is treated as 'nothing detected'.
    """
    return [v if MIN_SONAR < v < MAX_SONAR else MAX_SONAR for v in readings]


def sonar_zones(readings):
    """
    Collapse 16 raw sonar readings into 9 named directional zones.

    Rather than working with raw sonar indices (so3, so4 etc.) the rest
    of the code uses named zones — this makes the logic readable and
    means only this function needs updating if the sonar layout changes.

    Pioneer 3-DX front sonar mapping:
      so0 (-90 deg) = far right
      so1 (-50 deg) = right
      so2 (-30 deg) = front-right
      so3 (-10 deg) = front-right-centre
      so4 (+10 deg) = front-left-centre
      so5 (+30 deg) = front-left
      so6 (+50 deg) = left
      so7 (+90 deg) = far left
    """
    r = clean_readings(readings)
    return {
        'front':       min(r[3], r[4]),
        'front_wide':  min(r[2], r[3], r[4], r[5]),
        'front_left':  min(r[4], r[5], r[6]),
        'front_right': min(r[1], r[2], r[3]),
        'left':        min(r[6], r[7]),
        'right':       min(r[0], r[1]),
        'rear':        min(r[8], r[15]),
        'all_front':   min(r[1], r[2], r[3], r[4], r[5], r[6]),
        'side_min':    min(r[0], r[7]),
        'raw':         r,
    }


def obstacle_hits(robot_x, robot_y, robot_theta_rad, readings, grid_map):
    """
    Sonar-to-grid obstacle marking disabled.

    Angle offsets produce too much positional error at distance — a 5 degree
    angle error at 0.5m range moves the marked cell ~4cm off the actual wall.
    GPS-based marking via mark_obstacle_ahead() is used instead, which uses
    the robot's known GPS position and the actual measured sonar distance.
    Returns empty list always.
    """
    return []


def person_nearby(readings, threshold=THRESHOLD_PERSON):
    """
    Detect whether a person is close to the robot.

    Uses only the two front-centre sonars (so3 at -10 deg, so4 at +10 deg).
    Using wider sonars (so2-so5) caused false triggers from boxes as the
    robot passed within 60cm of them. The narrow two-sensor window means
    only something directly in front triggers a pause.
    """
    r = clean_readings(readings)
    return min(r[3], r[4]) < threshold


def predict_collision(robot_x, robot_y, robot_theta,
                      last_lv, last_rv, readings,
                      steps=8, timestep_s=0.064):
    """
    Predictive collision detection using differential drive kinematics.

    Simulates where the robot will be in 8 timesteps (512ms) by integrating:
        v     = (v_left + v_right) / 2         (forward speed)
        omega = (v_right - v_left) / axle_len  (turn rate)
        x += v * cos(theta) * dt
        y += v * sin(theta) * dt
        theta += omega * dt

    If the predicted position is within 0.15m of a measured obstacle,
    the collision is flagged before the robot reaches it.

    Returns (bool, str): (collision_predicted, direction_of_threat)
    """
    v_left  = last_lv * WHEEL_RADIUS
    v_right = last_rv * WHEEL_RADIUS
    v       = (v_left + v_right) / 2.0
    omega   = (v_right - v_left) / AXLE_LENGTH

    if v <= 0.05:
        return False, 'none'

    px, py, ptheta = robot_x, robot_y, robot_theta
    for _ in range(steps):
        px     += v * math.cos(ptheta) * timestep_s
        py     += v * math.sin(ptheta) * timestep_s
        ptheta += omega * timestep_s

    travel_dist = math.hypot(px - robot_x, py - robot_y)
    r = clean_readings(readings)

    front_min = min(r[2], r[3], r[4], r[5])
    if front_min < travel_dist + 0.15:
        return True, 'front'

    if omega > 0.5 and min(r[5], r[6], r[7]) < 0.25:
        return True, 'left'
    if omega < -0.5 and min(r[0], r[1], r[2]) < 0.25:
        return True, 'right'

    return False, 'none'


def collision_severity(zones):
    """
    Classify how serious the current obstacle situation is.

    This gives the AVOID state a graduated response — it doesn't
    treat a mild warning the same as being completely boxed in.

    Returns:
        'stuck'   — obstacles on all sides, aggressive recovery needed
        'danger'  — front or blind-spot critically close, must reverse
        'warning' — moderately close, steer away but keep moving
        'none'    — all clear, normal navigation
    """
    f    = zones['front_wide']
    l    = zones['left']
    r    = zones['right']
    side = zones['side_min']

    if (f    < THRESHOLD_DANGER
            and l < THRESHOLD_DANGER
            and r < THRESHOLD_DANGER):
        return 'stuck'

    if side < 0.14:
        return 'danger'

    if f < THRESHOLD_DANGER:
        return 'danger'

    if f < THRESHOLD_WARNING or l < 0.22 or r < 0.22:
        return 'warning'

    return 'none'