"""
6-DOF multi-object pick-and-place  -  SIMULATION and HARDWARE share the same brain.

    python pick_and_place.py                  # PyBullet simulation (default)
    python pick_and_place.py dryrun           # whole job through the HARDWARE code path, nothing connected
    python pick_and_place.py servo-test COM3  # one-time servo calibration (direction, zero, gripper pulses)
    python pick_and_place.py hardware COM3    # real arm + camera (needs vision_bridge.py + servo_controller.ino)

Architecture (this is what makes "swap simulation for hardware" a one-line change):

    +----------------------+      +--------------------------------------+
    |  Planner (pure maths)|      |  PickPlaceController                 |
    |  IK, paths, checks   |<-----|  the whole task logic; it only talks |
    +----------------------+      |  to the two small interfaces below   |
                                  +------------------+-------------------+
                                                     |
                  +----------------------------------+-----------------------------+
                  |  ARM  (move joints, gripper)     |  DETECTOR (where is the object)
        SimArm  <-+-> ServoArm (real servos)         |  SimDetector <-> CameraDetector (homography)
                                                     |  HintDetector (dry run, trusts typed positions)

Frame: robot base frame, origin on the table under the base axis, X forward, Y left, metres.
Joint zero = arm pointing straight up.  +q2/+q3/+q4 lean the arm toward +X, +q1 turns it counter-clockwise
seen from above, +q5 rolls the gripper about its own axis, as defined in arm6dof_v2.urdf.
All link dimensions are ESTIMATES copied from the URDF - re-measure the real arm and update them.

Glossary (the four lengths used everywhere):
    H_SHOULDER  height of the shoulder axis above the table
    L1          shoulder axis -> elbow axis        L2   elbow axis -> wrist-pitch axis
    D_TCP       wrist-pitch axis -> TCP.   TCP = "tool centre point" = the spot between the fingertips
                that has to be placed on the object.  IK answers: which joint angles put the TCP there?
"""
import math
import sys
import time
import numpy as np

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
URDF_PATH = r"C:\armproject\arm6dof_v2.urdf"

# --- arm geometry (must match the URDF; ESTIMATES until you measure the real arm) ---
H_SHOULDER = 0.085   # table -> shoulder axis
L1 = 0.105           # shoulder -> elbow
L2 = 0.100           # elbow -> wrist pitch axis
D_TCP = 0.140        # wrist pitch axis -> tool_center (0.045 + 0.095)

# Joint limits (rad) for j1..j5, taken from the SG1501MG travel of ~165 deg centred on zero.
# Replace with the real mechanical stops once the arm is built.  The simulation also reads the URDF.
JOINT_LIMITS = [(-1.44, 1.44)] * 5
GRIPPER_OPEN = 0.02          # gripper command range 0 (closed) .. 0.02 m per finger (open, ~50 mm gap)
GRIPPER_MIN_GAP = 0.010      # jaw gap when fully closed (from the URDF finger geometry)

# Gripper pitch from vertical (pi = straight down).  A straight-down grasp cannot lift clear of the
# cube with +/-82.5 deg joints, so the planner tries these in order and uses the first that fits the job.
PHI3_CANDIDATES = [math.pi, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3]

# --- task ---
CUBE_SIZE = 0.03            # 30 mm cubes (must be < 50 mm gripper opening)
GRASP_Z = CUBE_SIZE / 2     # TCP height at the cube centre
HOVER_Z = 0.055             # TCP height for the approach above an object / target
DROP_Z = GRASP_Z + 0.004    # TCP height when releasing
MIN_SAFE_Z = 0.045          # lowest TCP height allowed on joint-space moves
MIN_SPACING = 0.05          # min distance between objects / targets (m)
MAX_GRASP_ERROR = 0.010     # abort the grasp if the TCP is more than 10 mm off (when measurable)

# While CARRYING, the held cube must clear every other cube on the table (they are CUBE_SIZE tall).
# The planner uses the highest of these that the arm can reach; HOVER_Z is the last resort.
# (With HOVER_Z the held cube passes only ~10 mm above the others - one wobble and it knocks them over.)
CARRY_Z_CANDIDATES = [0.095, 0.085, 0.075, 0.065]
CARRY_WARN_Z = 0.070        # if only a low carry height is possible, warn about cubes close to the path
CARRY_WARN_DIST = 0.045     # ... closer than this (m)

# --- motion speeds (conservative; the same numbers protect the real servos) ---
V_JOINT = 1.2               # rad/s   joint-space moves
V_APPROACH = 0.04           # m/s     vertical approach / lift
V_TRANSIT = 0.10            # m/s     carrying between points
MIN_T_JOINT, MIN_T_LINE, MIN_T_TRANSIT = 1.5, 1.0, 1.5

# --- simulation only ---
GRASP_ASSIST = True         # lock the cube to the gripper once a grasp is VERIFIED (False = pure friction)
REALTIME, SPEED = True, 1.5
SIM_DT = 1.0 / 240.0

# --- camera / auto mode (hardware) ---
SLOT_PITCH = 0.065          # distance between drop slots when the camera mode places several objects (m)

# Demo scenario: object = (x, y, yaw_rad), target = (x, y), metres.
DEMO_JOBS = [
    {"object": (0.195, 0.113, 0.40), "target": (0.195, -0.113)},
    {"object": (0.235, 0.000, -0.30), "target": (0.125, -0.217)},
    {"object": (0.129, 0.184, 0.90), "target": (0.166, -0.166)},
]
COLORS = [[0.9, 0.2, 0.2, 1], [0.2, 0.4, 0.9, 1], [0.95, 0.75, 0.1, 1],
          [0.6, 0.2, 0.8, 1], [0.1, 0.7, 0.7, 1], [0.9, 0.5, 0.1, 1]]


# =============================================================================
# 2. KINEMATICS & PLANNING   (pure maths - no simulator, no hardware)
# =============================================================================
def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def smooth(s):
    return s * s * (3 - 2 * s)


def fk(q):
    """TCP position (x, y, z) from [q1, q2, q3, q4, ...]."""
    q1, q2, q3, q4 = q[:4]
    phi1, phi2, phi3 = q2, q2 + q3, q2 + q3 + q4
    r = L1 * math.sin(phi1) + L2 * math.sin(phi2) + D_TCP * math.sin(phi3)
    z = H_SHOULDER + L1 * math.cos(phi1) + L2 * math.cos(phi2) + D_TCP * math.cos(phi3)
    return (r * math.cos(q1), r * math.sin(q1), z)


def ik_arm(x, y, z, phi3, limits):
    """Analytic IK for joints 1-4 (TCP position + gripper pitch phi3). [q1..q4] or None."""
    q1 = math.atan2(y, x)
    r = math.hypot(x, y)
    rw = r - D_TCP * math.sin(phi3)                       # wrist-pitch point
    zw = z - D_TCP * math.cos(phi3) - H_SHOULDER
    c = (rw * rw + zw * zw - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1.0:
        return None
    q3 = math.acos(c)                                     # elbow-up solution
    q2 = math.atan2(rw, zw) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
    q4 = phi3 - q2 - q3
    q = [q1, q2, q3, q4]
    for qi, (lo, hi) in zip(q, limits[:4]):
        if qi < lo - 1e-9 or qi > hi + 1e-9:
            return None
    if H_SHOULDER + L1 * math.cos(q2) < 0.03:             # elbow must stay off the table
        return None
    return q


def roll_for(q1, phi3, obj_yaw, lim5):
    """Wrist roll q5 that lines the jaws up with a face of the object (any multiple of 90 deg)."""
    q5s = np.arange(lim5[0], lim5[1] + 1e-9, 0.0005)
    ang = q1 + np.arctan2(np.cos(q5s), -np.sin(q5s) * math.cos(phi3))   # horizontal jaw direction
    d = np.abs(((ang - obj_yaw) + math.pi / 4) % (math.pi / 2) - math.pi / 4)
    ok = np.where(d < 0.002)[0]
    if len(ok) == 0:
        return None
    return float(q5s[ok[np.argmin(np.abs(q5s[ok]))]])     # smallest roll that works


def joint_path(q0, q1, n):
    return [[a + (b - a) * smooth(k / n) for a, b in zip(q0, q1)] for k in range(1, n + 1)]


def line_path(xy, z0, z1, phi3, q5, limits, n):
    """Straight vertical TCP move; IK is solved along the whole line."""
    pts = []
    for k in range(1, n + 1):
        q = ik_arm(xy[0], xy[1], z0 + (z1 - z0) * smooth(k / n), phi3, limits)
        if q is None:
            return None
        pts.append(q + [q5])
    return pts


def transit_xy(xy0, xy1, s):
    """Table point a fraction s (0..1) of the way from xy0 to xy1 (polar interpolation, as the arm swings)."""
    r0, t0 = math.hypot(*xy0), math.atan2(xy0[1], xy0[0])
    r1, t1 = math.hypot(*xy1), math.atan2(xy1[1], xy1[0])
    r, t = r0 + (r1 - r0) * s, t0 + (t1 - t0) * s
    return r * math.cos(t), r * math.sin(t)


def transit_path(xy0, xy1, z, phi3, q5, limits, n):
    """Constant-height move between two table points (polar interpolation)."""
    pts = []
    for k in range(1, n + 1):
        x, y = transit_xy(xy0, xy1, smooth(k / n))
        q = ik_arm(x, y, z, phi3, limits)
        if q is None:
            return None
        pts.append(q + [q5])
    return pts


def min_tcp_z(path):
    return min(fk(q)[2] for q in path)


def plan_job(obj_xy, obj_yaw, tgt_xy, limits):
    """
    Returns (plan, None) or (None, reason).   plan = {"phi3", "q5", "carry_z"}
    The SAME gripper pitch and roll are used at the pick and at the target, so the object is carried
    and released flat (it is only turned about the vertical axis).
    carry_z = the highest TCP height (<= 95 mm) at which the whole carry is reachable.
    """
    zs_pick = np.linspace(GRASP_Z, HOVER_Z, 25)
    zs_drop = np.linspace(DROP_Z, HOVER_Z, 25)
    home = [0.0] * 5
    best = None                                              # the tilt that allows the HIGHEST carry wins
    pick_possible = place_possible = False
    for phi3 in PHI3_CANDIDATES:
        pk = all(ik_arm(*obj_xy, z, phi3, limits) for z in zs_pick)
        pl = all(ik_arm(*tgt_xy, z, phi3, limits) for z in zs_drop)
        pick_possible |= pk
        place_possible |= pl
        if not (pk and pl):
            continue
        qh_pick = ik_arm(*obj_xy, HOVER_Z, phi3, limits)
        qh_place = ik_arm(*tgt_xy, HOVER_Z, phi3, limits)
        q5 = roll_for(qh_pick[0], phi3, obj_yaw, limits[4])
        if q5 is None:
            continue
        if min_tcp_z(joint_path(home, qh_pick + [q5], 40)) < MIN_SAFE_Z:
            continue
        if min_tcp_z(joint_path(home, qh_place + [q5], 40)) < MIN_SAFE_Z:
            continue
        for cz in CARRY_Z_CANDIDATES + [HOVER_Z]:            # highest carry height that is reachable
            if not all(ik_arm(*obj_xy, z, phi3, limits) for z in np.linspace(GRASP_Z, cz, 25)):
                continue
            if not all(ik_arm(*tgt_xy, z, phi3, limits) for z in np.linspace(DROP_Z, cz, 25)):
                continue
            if transit_path(obj_xy, tgt_xy, cz, phi3, q5, limits, 40) is None:
                continue
            q_end = ik_arm(*tgt_xy, cz, phi3, limits) + [q5]
            if min_tcp_z(joint_path(q_end, home, 40)) < MIN_SAFE_Z:
                continue
            if best is None or cz > best["carry_z"] + 1e-9:   # strictly higher only: ties keep the straighter grasp
                best = {"phi3": phi3, "q5": q5, "carry_z": cz}
            break
    if best is not None:
        return best, None
    d_o, d_t = math.hypot(*obj_xy), math.hypot(*tgt_xy)
    a_o, a_t = math.degrees(math.atan2(obj_xy[1], obj_xy[0])), math.degrees(math.atan2(tgt_xy[1], tgt_xy[0]))
    if not pick_possible:
        return None, f"OBJECT is out of reach (distance {100*d_o:.1f} cm, angle {a_o:.0f} deg)"
    if not place_possible:
        return None, f"TARGET is out of reach (distance {100*d_t:.1f} cm, angle {a_t:.0f} deg)"
    return None, "object and target are each reachable, but not with one common gripper tilt / safe path"


def closest_to_path(xy0, xy1, pts):
    """Smallest distance (m) from the carry path xy0->xy1 to any of the given table points."""
    if not pts:
        return None
    path = [transit_xy(xy0, xy1, k / 60) for k in range(61)]
    return min(math.dist(a, b) for a in path for b in pts)


def reach_band(limits):
    """Smallest and largest table distance (m) where a grasp is possible (at angle 0)."""
    rs = [r for r in np.arange(0.12, 0.34, 0.005)
          if any(all(ik_arm(r, 0, z, ph, limits) for z in np.linspace(GRASP_Z, HOVER_Z, 15))
                 for ph in PHI3_CANDIDATES)]
    return (min(rs), max(rs)) if rs else (None, None)


def spacing_ok(jobs):
    """Objects and targets must be at least MIN_SPACING apart (an object and its own target excepted)."""
    pts = []
    for i, j in enumerate(jobs):
        pts.append((("object", i), j["object"][:2]))
        pts.append((("target", i), j["target"]))
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            (ka, ia), pa = pts[a]
            (kb, ib), pb = pts[b]
            if ia == ib and ka != kb:
                continue
            if math.dist(pa, pb) < MIN_SPACING:
                return f"{ka} {ia+1} and {kb} {ib+1} are closer than {100*MIN_SPACING:.0f} cm"
    return None


def drop_slots(first_xy, n, avoid=(), pitch=SLOT_PITCH):
    """Up to n drop positions on an arc (constant distance from the base), starting at first_xy and SLOT_PITCH apart.
    Positions closer than MIN_SPACING to any point in `avoid` (objects still on the table) are skipped."""
    r, a0 = math.hypot(*first_xy), math.atan2(first_xy[1], first_xy[0])
    sgn = 1 if a0 <= 0 else -1                            # walk toward the side with more room
    slots, k = [], 0
    while len(slots) < n and k < 60:
        a = a0 + sgn * k * pitch / r
        k += 1
        if abs(a) > math.radians(80):
            break
        p = (r * math.cos(a), r * math.sin(a))
        if any(math.dist(p, q) < MIN_SPACING for q in avoid):
            continue
        slots.append(p)
    return slots


# =============================================================================
# 3. THE TWO INTERFACES  (everything the controller needs - nothing more)
# =============================================================================
class ArmInterface:
    """
    What the controller needs from an arm.   dt = control period in seconds.
      joint_angles()          -> [q1..q5] last known angles (rad)
      command(q)              -> send joint targets and advance ONE control period (dt)
      wait(sec)               -> keep the last command and let time pass
      wait_reached(q, t)      -> wait until the joints are at q (or timeout); returns bool
      set_gripper(opening)    -> 0 (closed) .. GRIPPER_OPEN, one command for both jaws
      grasp_close_opening     -> value to command when closing on an object
      holding_object(width)   -> True / False if the jaws can tell, None if unknown
      tcp_position()          -> measured (x, y, z) or None if there is no measurement
      on_grasp() / on_release() -> hooks (simulation grasp-assist; no-ops on hardware)
      torque_report(q)        -> text about servo load, or ''
    """


class DetectorInterface:
    """
      locate(hint_xy, radius) -> (x, y, yaw) of the object nearest to hint_xy, or None
      locate_all()            -> [(x, y, yaw), ...] every object currently visible
    The hint is the position the USER gave; the detector returns the precise measured pose.
    """


# =============================================================================
# 4. THE CONTROLLER  (identical for simulation and hardware)
# =============================================================================
class PickPlaceController:
    def __init__(self, arm, detector, limits):
        self.arm, self.det, self.limits = arm, detector, limits

    # ---- motion primitives ----
    def _run(self, path):
        for q in path:
            self.arm.command(q)

    def move_joints(self, q_goal):
        q0 = self.arm.joint_angles()
        dur = max(MIN_T_JOINT, max(abs(a - b) for a, b in zip(q0, q_goal)) / V_JOINT)
        self._run(joint_path(q0, q_goal, max(2, int(dur / self.arm.dt))))
        self.arm.wait_reached(q_goal, 2.0)

    def move_line(self, xy, z0, z1, phi3, q5):
        dur = max(MIN_T_LINE, abs(z1 - z0) / V_APPROACH)
        path = line_path(xy, z0, z1, phi3, q5, self.limits, max(2, int(dur / self.arm.dt)))
        if path is None:
            raise RuntimeError(f"unreachable on the vertical line at ({xy[0]:.3f}, {xy[1]:.3f})")
        self._run(path)
        self.arm.wait_reached(path[-1], 2.0)

    def move_transit(self, xy0, xy1, phi3, q5, z=HOVER_Z):
        dur = max(MIN_T_TRANSIT, math.dist(xy0, xy1) / V_TRANSIT)
        path = transit_path(xy0, xy1, z, phi3, q5, self.limits, max(2, int(dur / self.arm.dt)))
        if path is None:
            raise RuntimeError("carry path is not reachable")
        self._run(path)
        self.arm.wait_reached(path[-1], 2.0)

    def go_home(self):
        self.move_joints([0.0] * 5)

    # ---- one job ----
    def run_job(self, n, job, others=()):
        """others = table points of cubes that are NOT part of this job (used only for a clearance warning)."""
        arm = self.arm
        print(f"\n--- Job {n} ---")
        hint, tgt = job["object"][:2], job["target"]

        # (1) detection: precise pose of the object the user pointed at
        obj = self.det.locate(hint, 0.06)
        if obj is None:
            print("SKIPPED: no object found near the given position")
            return "SKIPPED", "object not detected"
        ox, oy, oyaw = obj
        print(f"Detected object at ({100*ox:.1f}, {100*oy:.1f}) cm, yaw {math.degrees(oyaw):.0f} deg; "
              f"target ({100*tgt[0]:.1f}, {100*tgt[1]:.1f}) cm")

        # (2) reachability of object AND target
        plan, why = plan_job((ox, oy), oyaw, tgt, self.limits)
        if plan is None:
            print("SKIPPED:", why)
            return "SKIPPED", why
        phi3, q5, carry_z = plan["phi3"], plan["q5"], plan["carry_z"]
        print(f"Reachable. Gripper tilt {math.degrees(math.pi - phi3):.0f} deg from vertical, "
              f"wrist roll {math.degrees(q5):.0f} deg, carry height {1000*carry_z:.0f} mm")
        obj_xy = (ox, oy)
        if carry_z <= CARRY_WARN_Z:
            d = closest_to_path(obj_xy, tgt, list(others))
            if d is not None and d < CARRY_WARN_DIST:
                print(f"WARNING: only a low carry is possible and the path passes {1000*d:.0f} mm from another "
                      "cube - it may be knocked. Move that cube or the target.")
        qh_pick = ik_arm(*obj_xy, HOVER_Z, phi3, self.limits) + [q5]

        grasped = False
        error = None
        try:
            # (3) pick
            arm.set_gripper(GRIPPER_OPEN)
            self.move_joints(qh_pick)
            self.move_line(obj_xy, HOVER_Z, GRASP_Z, phi3, q5)
            arm.wait(0.3)
            tcp = arm.tcp_position()
            if tcp is not None:
                err = math.dist(tcp, (ox, oy, GRASP_Z))
                print(f"At grasp: TCP error {1000*err:.1f} mm")
                if err > MAX_GRASP_ERROR:
                    raise RuntimeError(f"arm did not reach the object (error {1000*err:.0f} mm) - "
                                       "check servo torque / link masses / calibration")
            rep = arm.torque_report(arm.joint_angles())
            if rep:
                print(rep)
            arm.set_gripper(arm.grasp_close_opening)
            arm.wait(1.0)
            held = arm.holding_object(CUBE_SIZE)
            print("Grip check:", {True: "object gripped", False: "NO OBJECT GRIPPED", None: "not measurable (assumed OK)"}[held])
            if held is False:
                raise RuntimeError("grasp failed (jaws closed on nothing)")
            arm.on_grasp()
            grasped = True
            self.move_line(obj_xy, GRASP_Z, carry_z, phi3, q5)
            if arm.holding_object(CUBE_SIZE) is False:
                raise RuntimeError("object dropped during lift")

            # (4) carry & place (same tilt and roll -> object stays flat, carried high above the other cubes)
            self.move_transit(obj_xy, tgt, phi3, q5, carry_z)
            self.move_line(tgt, carry_z, DROP_Z, phi3, q5)
            arm.on_release()
            grasped = False
            arm.set_gripper(GRIPPER_OPEN)
            arm.wait(1.0)
            self.move_line(tgt, DROP_Z, carry_z, phi3, q5)
        except Exception as e:
            error = str(e)
            print("FAILED:", error)
            arm.on_release()
            arm.set_gripper(GRIPPER_OPEN)
            arm.wait(0.5)

        # (5) safe home pose, then verify with the detector
        try:
            self.go_home()
            arm.wait(0.8)
        except Exception as e:
            print("Could not return home:", e)
        if error:
            return "FAILED", error
        res = self.det.locate(tgt, 0.06)
        if res is None:
            print("Placed (not verified: object not seen at the target)")
            return "PLACED", "unverified"
        e_mm = 1000 * math.dist(res[:2], tgt)
        ok = e_mm < 15
        print(f"Object ended at ({100*res[0]:.1f}, {100*res[1]:.1f}) cm, error {e_mm:.1f} mm -> "
              f"{'SUCCESS' if ok else 'MISSED THE TARGET'}")
        return ("SUCCESS" if ok else "MISSED"), f"{e_mm:.1f} mm"

    def run(self, jobs):
        summary = []
        for i, job in enumerate(jobs):
            # cubes still waiting to be moved + cubes already placed (their targets) are obstacles for this job
            others = [j["object"][:2] for j in jobs[i + 1:]] + [j["target"] for j in jobs[:i]]
            summary.append((i + 1, *self.run_job(i + 1, job, others)))
        print("\n=========== SUMMARY ===========")
        for n, status, info in summary:
            print(f"Job {n}: {status}  ({info})")
        return summary


# =============================================================================
# 5. SIMULATION BACKEND  (PyBullet)
# =============================================================================
class SimArm(ArmInterface):
    dt = SIM_DT
    grasp_close_opening = 0.0           # squeeze fully: a miss then reads 0, a 30 mm cube reads ~10 mm/jaw

    def __init__(self):
        import pybullet as p
        import pybullet_data
        self.p = p
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIM_DT)
        p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(0.75, 55, -40, [0.12, 0, 0.0])
        self.body = p.loadURDF(URDF_PATH, useFixedBase=True)

        self.idx = {p.getJointInfo(self.body, i)[1].decode(): i for i in range(p.getNumJoints(self.body))}
        names = ("j1_base", "j2_shoulder", "j3_elbow", "j4_wrist_pitch", "j5_wrist_roll")
        self.arm_idx = [self.idx[n] for n in names]
        self.grip, self.follow, self.tcp_idx = self.idx["j6_gripper"], self.idx["j6_gripper_follower"], self.idx["tcp_fixed"]
        self.gripper_link = self.idx["j5_wrist_roll"]           # link index == its parent joint's index
        self.limits = [(p.getJointInfo(self.body, i)[8], p.getJointInfo(self.body, i)[9]) for i in self.arm_idx]
        self.efforts = [p.getJointInfo(self.body, i)[10] for i in self.arm_idx]
        for name in ("j6_gripper", "j6_gripper_follower"):
            p.changeDynamics(self.body, self.idx[name], lateralFriction=1.5, spinningFriction=0.01)
        self._fix_masses()

        self.objects, self.constraint = [], None
        for i in self.arm_idx:
            p.resetJointState(self.body, i, 0.0)
        for i in (self.grip, self.follow):
            p.resetJointState(self.body, i, GRIPPER_OPEN)
        self.set_gripper(GRIPPER_OPEN)
        self.command([0.0] * 5)

    def _fix_masses(self):
        """
        A link WITHOUT an <inertial> block gets mass = 1 kg in PyBullet.  tool_center is such a link, so
        it used to put a 1 kg weight on the fingertips (needs 3.1 N.m at the shoulder; the servo gives 1.5)
        and the arm sagged to the floor.  Give it a realistic mass here; the URDF itself is untouched.
        """
        p = self.p
        p.changeDynamics(self.body, self.tcp_idx, mass=0.001, localInertiaDiagonal=[1e-7] * 3)
        total = 0.0
        for i in range(p.getNumJoints(self.body)):
            m = p.getDynamicsInfo(self.body, i)[0]
            total += m
            if m > 0.4:
                print(f"WARNING: link {p.getJointInfo(self.body, i)[12].decode()} has mass {m:.2f} kg "
                      "(missing <inertial> in the URDF?) - the arm may sag")
        print(f"Moving arm mass in simulation: {total:.2f} kg")

    # ---- ArmInterface ----
    def _step(self):
        self.p.stepSimulation()
        if REALTIME:
            time.sleep(SIM_DT / SPEED)

    def joint_angles(self):
        return [self.p.getJointState(self.body, i)[0] for i in self.arm_idx]

    def command(self, q):
        self.p.setJointMotorControlArray(self.body, self.arm_idx, self.p.POSITION_CONTROL,
                                         targetPositions=list(q), forces=self.efforts)
        self._step()
        self._last_q = list(q)

    def wait(self, sec):
        q = getattr(self, "_last_q", None)
        for _ in range(int(sec / SIM_DT)):
            if q is not None:
                self.p.setJointMotorControlArray(self.body, self.arm_idx, self.p.POSITION_CONTROL,
                                                 targetPositions=q, forces=self.efforts)
            self._step()

    def wait_reached(self, q, timeout=2.0):
        for _ in range(int(timeout / SIM_DT)):
            if max(abs(a - b) for a, b in zip(self.joint_angles(), q)) < 0.004:
                return True
            self.command(q)
        return False

    def set_gripper(self, opening):
        opening = min(max(opening, 0.0), GRIPPER_OPEN)
        for i in (self.grip, self.follow):                       # ONE control -> both jaws
            self.p.setJointMotorControl2(self.body, i, self.p.POSITION_CONTROL, targetPosition=opening, force=20)

    def holding_object(self, width):
        f1, f2 = (self.p.getJointState(self.body, i)[0] for i in (self.grip, self.follow))
        expected = width - GRIPPER_MIN_GAP                       # sum of both finger travels on contact
        print(f"  finger positions {1000*f1:.1f} / {1000*f2:.1f} mm (expected sum {1000*expected:.0f} mm)")
        return abs((f1 + f2) - expected) < 0.006 and min(f1, f2) > 0.002

    def tcp_position(self):
        return self.p.getLinkState(self.body, self.tcp_idx)[4]

    def on_grasp(self):
        if not GRASP_ASSIST or not self.objects:
            return
        p = self.p
        tcp = self.tcp_position()
        body = min(self.objects, key=lambda b: math.dist(p.getBasePositionAndOrientation(b)[0], tcp))
        cpos, corn = p.getBasePositionAndOrientation(body)
        if math.dist(cpos, tcp) > 0.035:
            return
        lpos, lorn = p.getLinkState(self.body, self.gripper_link)[4:6]
        inv = p.invertTransform(lpos, lorn)
        rel = p.multiplyTransforms(inv[0], inv[1], cpos, corn)
        self.constraint = p.createConstraint(self.body, self.gripper_link, body, -1, p.JOINT_FIXED, [0, 0, 0],
                                             parentFramePosition=rel[0], parentFrameOrientation=rel[1],
                                             childFramePosition=[0, 0, 0])

    def on_release(self):
        if self.constraint is not None:
            self.p.removeConstraint(self.constraint)
            self.constraint = None

    def torque_report(self, q):
        """How hard each servo works to hold this pose (static gravity load), as % of stall torque."""
        try:
            p = self.p
            f = [p.getJointState(self.body, i)[0] for i in (self.grip, self.follow)]
            tau = p.calculateInverseDynamics(self.body, list(q) + f, [0.0] * 7, [0.0] * 7)[:5]
            pct = [100 * abs(t) / e for t, e in zip(tau, self.efforts)]
            txt = "Holding load (% of servo stall): " + ", ".join(f"j{k+1} {v:.0f}%" for k, v in enumerate(pct))
            if max(pct) > 70:
                txt += "   <-- above 70%: real servos may overheat/sag"
            return txt
        except Exception:
            return ""

    # ---- scene helpers (simulation only) ----
    def draw_reach(self, band):
        if not band[0]:
            return
        for rr in band:
            pts = [[rr * math.cos(math.radians(a)), rr * math.sin(math.radians(a)), 0.001] for a in range(-80, 81, 4)]
            for a, b in zip(pts[:-1], pts[1:]):
                self.p.addUserDebugLine(a, b, [0.45, 0.45, 0.45], 2)

    def spawn_scene(self, jobs):
        p, h = self.p, CUBE_SIZE / 2
        for n, job in enumerate(jobs):
            x, y, yaw = job["object"]
            col = COLORS[n % len(COLORS)]
            cube = p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[h, h, h]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[h, h, h], rgbaColor=col),
                basePosition=[x, y, h + 0.001], baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]))
            p.changeDynamics(cube, -1, lateralFriction=1.5)
            self.objects.append(cube)
            tx, ty = job["target"]
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.0005],
                                                         rgbaColor=col[:3] + [0.6]),
                basePosition=[tx, ty, 0.0006])
            p.addUserDebugText(f"obj {n+1}", [x, y, 0.06], col[:3], textSize=1.1)
            p.addUserDebugText(f"target {n+1}", [tx, ty, 0.04], col[:3], textSize=1.1)
        self.wait(1.0)

    def idle(self):
        try:
            while self.p.isConnected():
                self._step()
        except self.p.error:
            pass


class SimDetector(DetectorInterface):
    """Stands in for camera + homography: reads true object poses from the simulator."""
    def __init__(self, sim):
        self.sim = sim

    def locate_all(self):
        p = self.sim.p
        out = []
        for b in self.sim.objects:
            pos, orn = p.getBasePositionAndOrientation(b)
            out.append((pos[0], pos[1], p.getEulerFromQuaternion(orn)[2]))
        return out

    def locate(self, hint_xy, radius=0.06):
        best, bd = None, radius
        for x, y, yaw in self.locate_all():
            d = math.dist((x, y), hint_xy)
            if d < bd:
                best, bd = (x, y, yaw), d
        return best


# =============================================================================
# 6. HARDWARE BACKEND  (real SR1501MG servos + Arduino + camera)
# =============================================================================
# HOW TO GO FROM SIMULATION TO THE REAL ARM  (do the steps in this order; each one is small)
#   1. python pick_and_place.py dryrun             -> the whole job through the hardware code, nothing connected
#   2. upload servo_controller.ino to the Arduino; servos on their OWN 5-6 V supply, grounds joined
#   3. python pick_and_place.py servo-test COM3    -> fixes sign + zero_us of every joint and the gripper pulses,
#                                                     then paste the printed lines into SERVO_CFG / GRIP_* below
#   4. measure the arm; update H_SHOULDER, L1, L2, D_TCP (top of this file) and the URDF
#   5. python vision_bridge.py calibrate / test    -> camera + table coordinates
#   6. set HW_CALIBRATED = True and run   python pick_and_place.py hardware COM3
HW_CALIBRATED = False        # set True only after step 3 (the program refuses to move the real arm before)

# Calibrate ONE joint at a time, with the arm in the HOME pose (straight up) first:
#   zero_us : pulse (us) that makes the joint look exactly like q = 0 in the URDF
#   sign    : +1 or -1 so that a POSITIVE q moves the joint the same way as in the simulation
# The kit's servos travel ~165 deg over 800..2200 us  ->  1400 us per 165 deg.
US_PER_RAD = 1400.0 / math.radians(165.0)
SERVO_CFG = [   # joint,           channel, sign, zero_us
    dict(name="j1_base",        ch=0, sign=+1, zero_us=1500),
    dict(name="j2_shoulder",    ch=1, sign=+1, zero_us=1500),
    dict(name="j3_elbow",       ch=2, sign=+1, zero_us=1500),
    dict(name="j4_wrist_pitch", ch=3, sign=+1, zero_us=1500),
    dict(name="j5_wrist_roll",  ch=4, sign=+1, zero_us=1500),
]
GRIP_CH = 5
GRIP_OPEN_US, GRIP_CLOSED_US = 1700, 1300      # pulses for fully open / fully closed jaws (calibrate!)
US_MIN, US_MAX = 800, 2200
HW_DT = 0.02                                    # 50 Hz control loop
HW_MAX_STEP = 1.5 * HW_DT                       # never move a joint faster than 1.5 rad/s
GRIP_STEP = 0.0015                              # gripper opening change per control tick (m)

JOINT_HINTS = {   # what a POSITIVE angle must do (same as in the simulation)
    "j1_base":        "+ turns the whole arm COUNTER-clockwise seen from above (toward the LEFT, +Y)",
    "j2_shoulder":    "+ leans the whole arm FORWARD (away from the base, toward +X)",
    "j3_elbow":       "+ folds the forearm FORWARD relative to the upper arm",
    "j4_wrist_pitch": "+ tips the gripper FORWARD relative to the forearm",
    "j5_wrist_roll":  "+ rolls the gripper counter-clockwise when you look at it from the fingertip end",
}


class SerialPulseWriter:
    """Sends 'S us0 us1 us2 us3 us4 us5\\n' to the servo controller (see servo_controller.ino)."""
    def __init__(self, port, baud=115200):
        try:
            import serial                                    # pip install pyserial
        except ImportError:
            raise SystemExit("pyserial is missing:  pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)                                  # Arduino resets when the port opens

    def __call__(self, pulses):
        self.ser.write(("S " + " ".join(str(int(round(u))) for u in pulses) + "\n").encode())

    def close(self):
        self.ser.close()


class DryRunWriter:
    """Stands in for the serial port: remembers what WOULD be sent to the servos."""
    def __init__(self):
        self.n, self.lo, self.hi, self.last, self.max_jump = 0, [10 ** 9] * 6, [0] * 6, None, 0.0

    def __call__(self, pulses):
        self.n += 1
        for i, u in enumerate(pulses):
            self.lo[i], self.hi[i] = min(self.lo[i], u), max(self.hi[i], u)
        if self.last is not None:
            self.max_jump = max(self.max_jump, max(abs(a - b) for a, b in zip(pulses[:5], self.last[:5])))   # joints only
        self.last = list(pulses)

    def close(self):
        pass

    def report(self):
        print("\n=== DRY-RUN servo report (what the real servos would have received) ===")
        names = [c["name"] for c in sorted(SERVO_CFG, key=lambda c: c["ch"])] + ["gripper"]
        for ch, name in enumerate(names):
            ang = ""
            if ch < 5:
                c = next(c for c in SERVO_CFG if c["ch"] == ch)
                a, b = [(u - c["zero_us"]) * c["sign"] / US_PER_RAD for u in (self.lo[ch], self.hi[ch])]
                ang = f"  = {math.degrees(min(a, b)):+.0f} .. {math.degrees(max(a, b)):+.0f} deg"
            print(f"  ch{ch} {name:15s} {self.lo[ch]:6.0f} .. {self.hi[ch]:6.0f} us{ang}")
        print(f"  {self.n} pulse packets; largest JOINT jump between two packets: {self.max_jump:.0f} us "
              f"= {math.degrees(self.max_jump / US_PER_RAD / HW_DT):.0f} deg/s")


class ServoArm(ArmInterface):
    dt = HW_DT
    grasp_close_opening = (CUBE_SIZE - GRIPPER_MIN_GAP) / 2 - 0.003    # squeeze slightly, don't stall the servo

    def __init__(self, write_pulses, limits=JOINT_LIMITS, realtime=True):
        self.write, self.limits, self.realtime = write_pulses, limits, realtime
        self._t = 0.0                                    # virtual clock for dry runs (realtime=False)
        self.q = [0.0] * 5                               # open loop: servos have no feedback.
        self.opening = GRIPPER_OPEN                      # START WITH THE ARM PHYSICALLY IN THE HOME POSE.
        self._send()

    # time (real, or virtual so that a dry run finishes in a second)
    def _now(self):
        return time.time() if self.realtime else self._t

    def _tick(self):
        if self.realtime:
            time.sleep(self.dt)
        else:
            self._t += self.dt

    def q_to_us(self, q):
        return [min(US_MAX, max(US_MIN, c["zero_us"] + c["sign"] * qi * US_PER_RAD)) for c, qi in zip(SERVO_CFG, q)]

    def grip_to_us(self, opening):
        f = min(max(opening / GRIPPER_OPEN, 0.0), 1.0)
        return GRIP_CLOSED_US + f * (GRIP_OPEN_US - GRIP_CLOSED_US)

    def _pulses(self):
        pulses = [0] * 6
        for c, u in zip(SERVO_CFG, self.q_to_us(self.q)):
            pulses[c["ch"]] = u
        pulses[GRIP_CH] = self.grip_to_us(self.opening)
        return pulses

    def _send(self):
        self.write(self._pulses())

    def joint_angles(self):
        return list(self.q)

    def command(self, q):
        new = []
        for qi, old, (lo, hi) in zip(q, self.q, self.limits):
            qi = min(max(qi, lo), hi)                                    # hard joint limits
            qi = old + max(-HW_MAX_STEP, min(HW_MAX_STEP, qi - old))    # speed limit
            new.append(qi)
        self.q = new
        self._send()
        self._tick()

    def wait(self, sec):
        end = self._now() + sec
        while self._now() < end:
            self._send()
            self._tick()

    def wait_reached(self, q, timeout=2.0):
        t_end = self._now() + timeout
        while max(abs(a - b) for a, b in zip(self.q, q)) > 1e-3 and self._now() < t_end:
            self.command(q)
        self.wait(0.3)                                   # let the servos finish (no encoder to ask)
        return True

    def set_gripper(self, opening):
        target = min(max(opening, 0.0), GRIPPER_OPEN)
        while abs(self.opening - target) > 1e-9:         # ramp the jaws instead of snapping them
            self.opening += max(-GRIP_STEP, min(GRIP_STEP, target - self.opening))
            self._send()
            self._tick()

    def holding_object(self, width):
        return None                                      # no sensor: add a current/force sensor to enable this

    def tcp_position(self):
        return None

    def on_grasp(self):
        pass

    def on_release(self):
        pass

    def torque_report(self, q):
        return ""


class CameraDetector(DetectorInterface):
    """
    The camera side.  vision_bridge.py (next to this file) builds one of these.
      detect_fn() -> list of (u, v, angle_rad) object detections in IMAGE pixels
      H           -> 3x3 homography (pixel -> table metres in the robot base frame; cv2.findHomography)
      r_range     -> ignore detections closer / farther than this from the base axis (the arm itself, the room)
    """
    def __init__(self, detect_fn, H, r_range=(0.10, 0.36)):
        self.detect_fn, self.H, self.r_range = detect_fn, np.asarray(H, float), r_range

    def _to_table(self, u, v):
        X, Y, W = self.H @ np.array([u, v, 1.0])
        return X / W, Y / W

    def locate_all(self):
        out = []
        for u, v, ang in self.detect_fn():
            x, y = self._to_table(u, v)
            if not (self.r_range[0] <= math.hypot(x, y) <= self.r_range[1]):
                continue
            x2, y2 = self._to_table(u + 10 * math.cos(ang), v + 10 * math.sin(ang))   # orientation via a 2nd point
            out.append((x, y, math.atan2(y2 - y, x2 - x)))
        return out

    def locate(self, hint_xy, radius=0.06):
        best, bd = None, radius
        for x, y, yaw in self.locate_all():
            d = math.dist((x, y), hint_xy)
            if d < bd:
                best, bd = (x, y, yaw), d
        return best


class HintDetector(DetectorInterface):
    """Dry-run stand-in for the camera: trusts the object positions the user typed."""
    def __init__(self, jobs):
        self.objs = [tuple(j["object"]) for j in jobs]

    def locate_all(self):
        return list(self.objs)

    def locate(self, hint_xy, radius=0.06):
        near = [o for o in self.objs if math.dist(o[:2], hint_xy) < radius]
        return min(near, key=lambda o: math.dist(o[:2], hint_xy)) if near else None


def servo_test(port):
    """Interactive one-time calibration. Moves ONE joint a few degrees at a time; you only answer questions."""
    print("\n=== SERVO TEST ===\nHOME pose = arm straight up. Keep your hand near the supply switch.\n"
          "Servos move only about 10 degrees each time, slowly.")
    input("Put the arm in the HOME pose, switch the servo supply on, press Enter... ")
    arm = ServoArm(SerialPulseWriter(port))
    arm.wait(1.0)
    try:
        while True:
            s = input("\nTest which joint?  1-5 = joints, g = gripper, q = quit and print the settings: ").strip().lower()
            if s == "q":
                break
            if s == "g":
                _gripper_test(arm)
            elif s in ("1", "2", "3", "4", "5"):
                _joint_test(arm, int(s) - 1)
    finally:
        arm.wait(0.2)
        print("\n=== paste these lines into pick_and_place.py (section 6) ===")
        print("SERVO_CFG = [")
        for c in SERVO_CFG:
            print(f"    dict(name=\"{c['name']}\", ch={c['ch']}, sign={c['sign']:+d}, zero_us={c['zero_us']}),")
        print("]")
        print(f"GRIP_OPEN_US, GRIP_CLOSED_US = {GRIP_OPEN_US}, {GRIP_CLOSED_US}")
        print("Then run every joint through its full range by hand-checking the mechanical stops, and set "
              "JOINT_LIMITS and HW_CALIBRATED = True.")
        arm.write.close()


def _joint_test(arm, k):
    cfg = SERVO_CFG[k]
    print(f"\n{cfg['name']}:  {JOINT_HINTS[cfg['name']]}")
    while True:
        base, target = arm.q[k], arm.q[k] + math.radians(10)
        for i in range(1, 101):                                    # 10 deg in 2 s
            q = list(arm.q)
            q[k] = base + (target - base) * i / 100
            arm.command(q)
        arm.wait(0.5)
        ans = input("Did it move THAT way (as described above)? [y/n/r = repeat]: ").strip().lower()
        for i in range(1, 101):                                    # back
            q = list(arm.q)
            q[k] = target + (base - target) * i / 100
            arm.command(q)
        if ans == "n":
            cfg["sign"] = -cfg["sign"]
            print(f"  -> sign flipped to {cfg['sign']:+d}; testing again")
            continue
        if ans != "r":
            break
    print("Now check the HOME position of this joint (straight, exactly like q = 0 in the simulation).")
    while True:
        s = input(f"  zero_us is {cfg['zero_us']}. Type microseconds to add (e.g. 15 or -20), Enter = it is straight: ").strip()
        if not s:
            break
        try:
            cfg["zero_us"] += int(s)
        except ValueError:
            continue
        arm._send()
        arm.wait(0.5)
    print(f"  {cfg['name']}: sign={cfg['sign']:+d}, zero_us={cfg['zero_us']}")


def _gripper_test(arm):
    global GRIP_OPEN_US, GRIP_CLOSED_US
    print("\nGripper: type a pulse in microseconds (800-2200) to move it; Enter = done.\n"
          "Find OPEN (jaws ~50 mm apart) and CLOSED (jaws just touching, servo NOT straining).")
    while True:
        s = input("  pulse us: ").strip()
        if not s:
            break
        try:
            u = min(US_MAX, max(US_MIN, int(s)))
        except ValueError:
            continue
        pulses = arm._pulses()
        pulses[GRIP_CH] = u
        arm.write(pulses)
        arm.wait(0.5)
        arm.write(pulses)
    for label in ("OPEN", "CLOSED"):
        s = input(f"  Pulse that gives fully {label} jaws (Enter = keep): ").strip()
        if s.isdigit():
            if label == "OPEN":
                GRIP_OPEN_US = int(s)
            else:
                GRIP_CLOSED_US = int(s)
    arm._send()


# =============================================================================
# 7. USER INPUT & MAIN
# =============================================================================
def _ask_xy(prompt):
    while True:
        try:
            v = [float(t) for t in input(prompt).replace(",", " ").split()]
            if len(v) == 2:
                return v[0] / 100, v[1] / 100
        except ValueError:
            pass
        except EOFError:
            return None
        print("  Please type two numbers in cm, like:  20 -11")


def auto_jobs(detector, limits):
    """Camera finds the objects, the user only chooses where they go (several objects -> slots along an arc)."""
    found = detector.locate_all()
    if not found:
        print("The camera sees no objects (check lighting, vision_bridge.py settings, calibration).")
        return []
    print(f"The camera sees {len(found)} object(s):")
    for i, (x, y, yaw) in enumerate(found, 1):
        print(f"  {i}: x={100*x:.1f}  y={100*y:.1f} cm   yaw {math.degrees(yaw):.0f} deg")
    print("(Every visible object is moved. Take objects that are already where you want them off the table first.)")
    while True:
        first = _ask_xy("Where should the FIRST object go?  x y in cm (the others go beside it along an arc): ")
        if first is None:
            return []
        if any(math.dist(first, o[:2]) < MIN_SPACING for o in found):
            print(f"  that spot is closer than {100*MIN_SPACING:.0f} cm to an object on the table - choose another")
            continue
        break
    keep = sorted(found, key=lambda o: math.dist(o[:2], first))
    slots = drop_slots(first, len(keep), [o[:2] for o in found])
    if len(slots) < len(keep):
        print(f"  only {len(slots)} free slot(s) fit on the arc: the other {len(keep) - len(slots)} object(s) stay where they are")
    jobs = []
    for o, s in zip(keep, slots):
        job = {"object": o, "target": s}
        msg = spacing_ok(jobs + [job])
        if msg:
            print(f"  skipping object at ({100*o[0]:.1f}, {100*o[1]:.1f}) cm: {msg}")
            continue
        plan, why = plan_job(o[:2], o[2], s, limits)
        print(f"  object ({100*o[0]:.1f}, {100*o[1]:.1f}) -> slot ({100*s[0]:.1f}, {100*s[1]:.1f}): "
              + ("reachable" if plan else f"NOT reachable, will be skipped: {why}"))
        jobs.append(job)
    return jobs


def ask_jobs(limits, band, backend="sim", detector=None):
    print("\n=== Pick-and-place task ===")
    if band[0]:
        print(f"Reachable ring (estimated model): {100*band[0]:.0f} - {100*band[1]:.0f} cm from the base axis, "
              "within +/-80 deg of the +X axis.")
        print("(X forward from the base, Y to the left.  Object and target should be at a similar distance.)")
    if backend in ("sim", "dryrun"):
        try:
            ans = input("Use the demo scenario with 3 objects? [Y/n]: ").strip().lower()
        except EOFError:
            ans = "y"
        if ans in ("", "y", "yes"):
            return [dict(j) for j in DEMO_JOBS]
    elif detector is not None:
        try:
            mode = input("[1] camera finds the objects, you choose where they go   "
                         "[2] you type object AND target positions: ").strip()
        except EOFError:
            mode = "1"
        if mode in ("", "1"):
            jobs = auto_jobs(detector, limits)
            if jobs:
                return jobs
    jobs = []
    while len(jobs) < len(COLORS):
        n = len(jobs) + 1
        try:
            s = input(f"\nObject {n}: x y [yaw_deg] in cm  (Enter = finish): ").strip()
            if not s:
                break
            v = [float(t) for t in s.replace(",", " ").split()]
            if len(v) not in (2, 3):
                raise ValueError
            obj = (v[0] / 100, v[1] / 100, math.radians(v[2]) if len(v) == 3 else 0.0)
            t = [float(t) for t in input(f"Target for object {n}: x y in cm: ").replace(",", " ").split()]
            if len(t) != 2:
                raise ValueError
        except (ValueError, EOFError):
            print("  Please type numbers like:  20 11 25   (x y yaw)  /  20 -11   (x y)")
            continue
        job = {"object": obj, "target": (t[0] / 100, t[1] / 100)}
        msg = spacing_ok(jobs + [job])
        if msg:
            print("  Rejected:", msg)
            continue
        plan, why = plan_job(obj[:2], obj[2], job["target"], limits)
        print("  -> reachable, will be done" if plan else f"  -> NOT reachable, will be skipped: {why}")
        jobs.append(job)
    if backend == "hardware":
        return jobs
    return jobs or [dict(j) for j in DEMO_JOBS]


def load_camera():
    try:
        import vision_bridge
    except ImportError:
        raise SystemExit("Hardware mode needs vision_bridge.py next to this script (it provides make_detector()).")
    return vision_bridge.make_detector()


def main():
    args = sys.argv[1:]
    backend = args[0].lower() if args else "sim"
    port = args[1] if len(args) > 1 else "COM3"

    if backend == "servo-test":
        servo_test(port)
        return

    detector = None
    if backend == "sim":
        sim = SimArm()
        for a, b in zip(sim.limits, JOINT_LIMITS):
            if abs(a[0] - b[0]) > 1e-6 or abs(a[1] - b[1]) > 1e-6:
                print("NOTE: URDF joint limits differ from JOINT_LIMITS; using the URDF values.")
                break
        limits, arm = sim.limits, sim
    elif backend == "dryrun":
        limits, arm = JOINT_LIMITS, ServoArm(DryRunWriter(), realtime=False)
    elif backend == "hardware":
        if not HW_CALIBRATED:
            raise SystemExit("HW_CALIBRATED is False. Run  python pick_and_place.py servo-test " + port +
                             "  first, paste its output into section 6 and set HW_CALIBRATED = True.\n"
                             "(Wrong servo directions or zeros can drive the arm into the table.)")
        detector = load_camera()                              # camera first: the arm is still still
        input("Put the arm in the HOME pose (straight up), clear the table, switch on the servo supply, "
              "then press Enter...")
        limits, arm = JOINT_LIMITS, ServoArm(SerialPulseWriter(port))
    else:
        raise SystemExit("usage: python pick_and_place.py [sim | dryrun | servo-test COMx | hardware COMx]")

    band = reach_band(limits)
    if backend == "sim":
        arm.draw_reach(band)
    jobs = ask_jobs(limits, band, backend, detector)
    if not jobs:
        raise SystemExit("No jobs to do.")
    msg = spacing_ok(jobs)
    if msg:
        raise SystemExit("Scenario rejected: " + msg)

    if backend == "sim":
        arm.spawn_scene(jobs)
        detector = SimDetector(arm)
    elif backend == "dryrun":
        detector = HintDetector(jobs)

    try:
        PickPlaceController(arm, detector, limits).run(jobs)
    except KeyboardInterrupt:
        print("\nSTOPPED by user. The servos keep holding their last pose - switch off the servo supply if needed.")
    finally:
        if backend == "dryrun":
            arm.write.report()
        elif backend == "hardware":
            arm.write.close()
    if backend == "sim":
        arm.idle()


if __name__ == "__main__":
    main()