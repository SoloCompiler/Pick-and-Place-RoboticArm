import math
import numpy as np

URDF_PATH = r"D:/College/Projects or Scripts/Robotic Pick and Place/arm6dof_v2.urdf"
# URDF_PATH = r"C:\armproject\arm6dof_v2.urdf"

H_SHOULDER = 0.085
L1 = 0.105
L2 = 0.100
D_TCP = 0.140

JOINT_LIMITS = [(-1.44, 1.44)] * 5
GRIPPER_OPEN = 0.02
GRIPPER_MIN_GAP = 0.010

PHI3_CANDIDATES = [math.pi, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3]

CUBE_SIZE = 0.03
GRASP_Z = CUBE_SIZE / 2
HOVER_Z = 0.055
DROP_Z = GRASP_Z + 0.004
MIN_SAFE_Z = 0.045
MIN_SPACING = 0.05
MAX_GRASP_ERROR = 0.010

CARRY_Z_CANDIDATES = [0.095, 0.085, 0.075, 0.065]
CARRY_WARN_Z = 0.070
CARRY_WARN_DIST = 0.045

V_JOINT = 1.2
V_APPROACH = 0.04
V_TRANSIT = 0.10
MIN_T_JOINT, MIN_T_LINE, MIN_T_TRANSIT = 1.5, 1.0, 1.5

GRASP_ASSIST = True
REALTIME, SPEED = True, 1.5
SIM_DT = 1.0 / 240.0

SLOT_PITCH = 0.065

DEMO_JOBS = [
    {"object": (0.195, 0.113, 0.40), "target": (0.195, -0.113)},
    {"object": (0.235, 0.000, -0.30), "target": (0.125, -0.217)},
    {"object": (0.129, 0.184, 0.90), "target": (0.166, -0.166)},
]
COLORS = [[0.9, 0.2, 0.2, 1], [0.2, 0.4, 0.9, 1], [0.95, 0.75, 0.1, 1],
          [0.6, 0.2, 0.8, 1], [0.1, 0.7, 0.7, 1], [0.9, 0.5, 0.1, 1]]


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def smooth(s):
    return s * s * (3 - 2 * s)


def fk(q):
    q1, q2, q3, q4 = q[:4]
    phi1, phi2, phi3 = q2, q2 + q3, q2 + q3 + q4
    r = L1 * math.sin(phi1) + L2 * math.sin(phi2) + D_TCP * math.sin(phi3)
    z = H_SHOULDER + L1 * math.cos(phi1) + L2 * math.cos(phi2) + D_TCP * math.cos(phi3)
    return (r * math.cos(q1), r * math.sin(q1), z)


def ik_arm(x, y, z, phi3, limits):
    q1 = math.atan2(y, x)
    r = math.hypot(x, y)
    rw = r - D_TCP * math.sin(phi3)
    zw = z - D_TCP * math.cos(phi3) - H_SHOULDER
    c = (rw * rw + zw * zw - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1.0:
        return None
    q3 = math.acos(c)
    q2 = math.atan2(rw, zw) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
    q4 = phi3 - q2 - q3
    q = [q1, q2, q3, q4]
    for qi, (lo, hi) in zip(q, limits[:4]):
        if qi < lo - 1e-9 or qi > hi + 1e-9:
            return None
    if H_SHOULDER + L1 * math.cos(q2) < 0.03:
        return None
    return q


def roll_for(q1, phi3, obj_yaw, lim5):
    q5s = np.arange(lim5[0], lim5[1] + 1e-9, 0.0005)
    ang = q1 + np.arctan2(np.cos(q5s), -np.sin(q5s) * math.cos(phi3))
    d = np.abs(((ang - obj_yaw) + math.pi / 4) % (math.pi / 2) - math.pi / 4)
    ok = np.where(d < 0.002)[0]
    if len(ok) == 0:
        return None
    return float(q5s[ok[np.argmin(np.abs(q5s[ok]))]])


def joint_path(q0, q1, n):
    return [[a + (b - a) * smooth(k / n) for a, b in zip(q0, q1)] for k in range(1, n + 1)]


def line_path(xy, z0, z1, phi3, q5, limits, n):
    pts = []
    for k in range(1, n + 1):
        q = ik_arm(xy[0], xy[1], z0 + (z1 - z0) * smooth(k / n), phi3, limits)
        if q is None:
            return None
        pts.append(q + [q5])
    return pts


def transit_xy(xy0, xy1, s):
    r0, t0 = math.hypot(*xy0), math.atan2(xy0[1], xy0[0])
    r1, t1 = math.hypot(*xy1), math.atan2(xy1[1], xy1[0])
    r, t = r0 + (r1 - r0) * s, t0 + (t1 - t0) * s
    return r * math.cos(t), r * math.sin(t)


def transit_path(xy0, xy1, z, phi3, q5, limits, n):
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
    zs_pick = np.linspace(GRASP_Z, HOVER_Z, 25)
    zs_drop = np.linspace(DROP_Z, HOVER_Z, 25)
    home = [0.0] * 5
    best = None
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
        for cz in CARRY_Z_CANDIDATES + [HOVER_Z]:
            if not all(ik_arm(*obj_xy, z, phi3, limits) for z in np.linspace(GRASP_Z, cz, 25)):
                continue
            if not all(ik_arm(*tgt_xy, z, phi3, limits) for z in np.linspace(DROP_Z, cz, 25)):
                continue
            if transit_path(obj_xy, tgt_xy, cz, phi3, q5, limits, 40) is None:
                continue
            q_end = ik_arm(*tgt_xy, cz, phi3, limits) + [q5]
            if min_tcp_z(joint_path(q_end, home, 40)) < MIN_SAFE_Z:
                continue
            if best is None or cz > best["carry_z"] + 1e-9:
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
    if not pts:
        return None
    path = [transit_xy(xy0, xy1, k / 60) for k in range(61)]
    return min(math.dist(a, b) for a in path for b in pts)


def reach_band(limits):
    rs = [r for r in np.arange(0.12, 0.34, 0.005)
          if any(all(ik_arm(r, 0, z, ph, limits) for z in np.linspace(GRASP_Z, HOVER_Z, 15))
                 for ph in PHI3_CANDIDATES)]
    return (min(rs), max(rs)) if rs else (None, None)


def spacing_ok(jobs):
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
    r, a0 = math.hypot(*first_xy), math.atan2(first_xy[1], first_xy[0])
    sgn = 1 if a0 <= 0 else -1
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