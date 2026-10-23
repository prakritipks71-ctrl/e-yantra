'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2A of Niti Vahan (NV) Theme.
*
*  Filename:         task2a.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         task2a.py
# Functions:        detect_lane, compute_control
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2A.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# The integer signal the evaluation raises when the vehicle must stop.
STOP_SIGNAL = 'NV_stop'
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_lane(frame):
    '''
    Purpose:
    ---
    Work out, from a single camera frame, how wrong the vehicle currently is -
    and hand back whatever `compute_control()` needs in order to correct it.

    This is your Task 1C detector, extended. Task 1C answered *where the lane
    centre is* and *which lane the vehicle is in*; a controller cannot use
    either of those directly. It needs to know how far the vehicle is from
    where it should be, and which way it is pointing.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        One BGR frame from the vehicle's camera, of shape (height, width, 3),
        the right way up and in the right colour order.

    Returns:
    ---
    Anything you like - a tuple, a dictionary, your own class. Whatever you
    return here is passed straight to compute_control() and nowhere else, so
    the two functions only have to agree with each other.

    Example call:
    ---
    detection = detect_lane(frame)

    REMEMBER:
    ---
    Both lanes are driven during evaluation, and the arena looks mirrored from
    the other one. Do not hard-code which side the boundary is on.

    Do NOT call cv2.imshow() or cv2.imwrite() from inside this function.

    You are marked on how closely you hold the MIDDLE of your lane, not merely
    on staying the correct side of the dashed line.
    '''

    # ADD YOUR CODE HERE
    import math

    # state kept between frames, stored on the function itself
    S = detect_lane.__dict__.setdefault(
        "S", {"side": None, "width": {"near": None, "far": None}})

    REF_X_FRAC = 0.5       # where the vehicle axis is in the image - check this
    LANE_WIDTH_M = 0.2     # lane spacing from Task 1B

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 90, 90), (40, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    def peaks(mask, y1, y2, min_px=3):
        hist = np.sum(mask[y1:y2, :] > 0, axis=0).astype(np.float32)
        hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 9), 0).ravel()
        if hist.max() < min_px:
            return []
        xs = np.where(hist >= max(min_px, 0.3 * hist.max()))[0]
        groups = np.split(xs, np.where(np.diff(xs) > 6)[0] + 1)
        return [float(np.average(g, weights=hist[g])) for g in groups if len(g) >= 2]

    def band_centre(name, ys, ws, ref):
        side, width = S["side"], S["width"][name]
        if ys and ws:
            y = min(ys, key=lambda x: abs(x - ref))      # my yellow edge
            wx = min(ws, key=lambda x: abs(x - y))       # divider next to it
            d = wx - y
            ok = abs(d) > 10 and (width is None or 0.5 * width < abs(d) < 1.6 * width)
            if ok and (side is None or np.sign(d) == side):
                S["side"] = int(np.sign(d))
                S["width"][name] = abs(d) if width is None else 0.9 * width + 0.1 * abs(d)
                return (y + wx) / 2                      # MIDDLE of the lane
        if side is None or width is None:
            return None
        if ys:                                           # dashed line in a gap
            y = min(ys, key=lambda x: abs(x - ref))
            return y + side * width / 2
        if ws:                                           # yellow not visible
            wx = min(ws, key=lambda x: abs(x - ref))
            return wx - side * width / 2
        return None

    ref = REF_X_FRAC * w
    near = (int(h * 0.75), int(h * 0.95))
    far = (int(h * 0.55), int(h * 0.72))

    near_c = band_centre("near", peaks(yellow, *near), peaks(white, *near), ref)
    if near_c is None:
        return {"valid": False}
    far_c = band_centre("far", peaks(yellow, *far), peaks(white, *far), ref)
    if far_c is None:
        far_c = near_c

    width_px = S["width"]["near"] or (w * 0.4)
    e_m = (near_c - ref) / width_px * LANE_WIDTH_M       # + : lane centre is to the right
    psi = math.atan2(far_c - near_c, h * (0.85 - 0.635)) # + : lane bends to the right

    return {"valid": True, "e_m": float(e_m), "psi": float(psi)}





def compute_control(detection, stop_requested):
    '''
    Purpose:
    ---
    Turn the errors from detect_lane() into the three numbers the vehicle is
    driven with, and obey the stop flag.

    This is where your Task 1B controller and your Task 1A steering geometry
    meet: work out one steering angle from the error, then split it into the
    two front-wheel angles the joints actually take.

    Input Arguments:
    ---
    `detection` :   [ any ]
        Whatever detect_lane() returned for this frame.

    `stop_requested` :  [ bool ]
        True while the evaluation is holding the vehicle at a stop point.
        This is a LEVEL, not a pulse: it stays True for the whole duration of
        the stop and goes False when the wait is over. While it is True the
        vehicle must be STATIONARY, so return a speed of 0.0. There is no
        stopwatch for you to write and no duration to hard-code.

    Returns:
    ---
    `left_angle`, `right_angle` :   [ float ]
        The two front-wheel steering angles, in RADIANS, written straight to
        the steering joints. The scene's steering joints are POSITIVE TO THE
        LEFT. Nothing is clamped for you - the vehicle's own limit is 45
        degrees, and commanding past it will not steer any harder.

    `speed` :   [ float ]
        Target velocity for both drive motors, in rad/s. 0.0 holds the
        vehicle still.

    Example call:
    ---
    left_angle, right_angle, speed = compute_control(detection, stop_requested)

    REMEMBER:
    ---
    Keep the loop running while stopped - set the SPEED to zero, not your
    program. If you stop reading frames you will not see the flag drop.

    Sooner or later a frame comes back with nothing usable in it. Holding the
    last steering angle, easing off the throttle and stopping dead are all
    defensible - doing nothing is not.
    '''

        # ADD YOUR CODE HERE
    import math

    K_HEAD, K_CTE = 0.8, 4.0
    CRUISE, SLOW = 0.8, 0.4
    MAX_STEER = math.radians(30)
    WHEELBASE, TRACK_WIDTH = 0.120, 0.110

    st = compute_control.__dict__.setdefault("st", {"steer": 0.0})

    if detection and detection.get("valid"):
        e, psi = detection["e_m"], detection["psi"]
        # steering is positive to the LEFT, so "lane is to the right" -> negative
        delta = -(K_HEAD * psi + math.atan(K_CTE * e / (CRUISE * 0.1 + 0.15)))
        delta = max(-MAX_STEER, min(MAX_STEER, delta))
        st["steer"] = 0.7 * st["steer"] + 0.3 * delta
        speed = CRUISE
    else:
        speed = SLOW                    # lane lost: keep last steering, slow down

    # Ackermann split (swap in your Task 1A function if you like)
    d = st["steer"]
    if abs(d) < 1e-6:
        left_angle = right_angle = 0.0
    else:
        r = WHEELBASE / math.tan(abs(d))
        inner = math.atan(WHEELBASE / max(1e-3, r - TRACK_WIDTH / 2))
        outer = math.atan(WHEELBASE / (r + TRACK_WIDTH / 2))
        left_angle, right_angle = (inner, outer) if d > 0 else (-outer, -inner)

    if stop_requested:
        speed = 0.0

    return float(left_angle), float(right_angle), float(speed)
    return 0.0, 0.0, 0.0


##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera and writing to the joints.
# None of it involves a decision you are being marked on - it is here so that
# every team reads the same picture and drives the same vehicle.
#
#####################################################################

def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.

    Each getObject() is a network round trip, which is why they all happen
    here rather than inside the loop.
    '''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2A.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Open Task_2A.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    The vision sensor hands back a raw byte buffer and a resolution, not an
    image, and three corrections are needed to turn one into the other. None
    of these problems announces itself - each produces a detector that looks
    like it is working on a frame that is wrong:

      * handleVisionSensor() - this scene's sensor renders only when asked.
        Without it every read returns the SAME frame for the whole lap.
      * cv2.flip(frame, 0) - the buffer's first row is the BOTTOM of the
        image, so the road comes out vertically mirrored.
      * RGB2BGR - the buffer is RGB and OpenCV assumes BGR. Without the swap
        every HSV threshold you tuned in Task 1C selects the wrong colour.

    Note `res[1], res[0]` in the reshape too: the resolution comes back as
    (width, height) while NumPy wants rows first.

    Returns None if no frame is ready yet, which happens on the first few
    iterations after the simulation starts.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Write the two steering angles and the drive speed to the joints.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, places the vehicle in the lane being tested
    # and starts the simulation - so wait for that rather than starting your
    # own, which would race the harness into position.
    #
    # Your script must never call startSimulation(), stopSimulation() or move
    # the vehicle. Reading the camera and driving the joints is all it does.
    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        # The harness stops the simulation when the lap is over, which is the
        # cue to exit - hence testing the state rather than looping forever.
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            # getInt32Signal() returns None until the signal is first set, so
            # "not set" reads as "not stopping" and the first frames behave.
            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))

            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection,
                                                             stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        # A call raising because the simulation stopped mid-frame is normal at
        # the end of a run; anything else is a bug worth seeing.
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        # Without this, whatever speed was last commanded is still being
        # applied and the vehicle drives on unsteered.
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
