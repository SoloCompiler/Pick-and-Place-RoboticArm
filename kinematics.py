"""Forward and inverse kinematics for the 6-DOF arm (arm6dof_v2.urdf).

Zero pose = arm pointing straight up. Units: metres and radians.
Joints q2, q3, q4 all rotate about Y, so the arm lives in one vertical
plane. q1 (base yaw) rotates that plane about the vertical axis.
"""
import math

# ESTIMATED from the product photo. Update after measuring the real arm.
H = 0.085    # table to shoulder axis (0.045 + 0.04)
L1 = 0.105   # shoulder to elbow
L2 = 0.100   # elbow to wrist-pitch axis
D = 0.140    # wrist-pitch axis to tool_center (0.045 + 0.095)
LIM = 1.44   # joint limit, +/- rad


def fk_planar(q2, q3, q4):
    """Return (elbow, wrist, tcp), each an (r, z) tuple.
    r = horizontal distance from base axis, z = height above table."""
    phi1 = q2                 # angle of upper arm from vertical
    phi2 = q2 + q3            # angle of forearm from vertical
    phi3 = q2 + q3 + q4       # angle of gripper from vertical

    # a link at angle phi with length L points along (sin phi, cos phi)
    elbow = (L1 * math.sin(phi1), H + L1 * math.cos(phi1))
    wrist = (elbow[0] + L2 * math.sin(phi2), elbow[1] + L2 * math.cos(phi2))
    tcp = (wrist[0] + D * math.sin(phi3), wrist[1] + D * math.cos(phi3))
    return elbow, wrist, tcp


def fk_3d(q1, q2, q3, q4):
    """Return the tool centre point (x, y, z) in the base frame."""
    _, _, (r, z) = fk_planar(q2, q3, q4)
    return r * math.cos(q1), r * math.sin(q1), z


def ik(x, y, z, phi3=math.pi, obj_angle=0.0):
    """Return (q1, q2, q3, q4, q5), or None if unreachable / out of limits.

    phi3 = desired gripper angle from vertical (pi = pointing straight down).
    obj_angle = object orientation about the vertical axis in the base frame.
    Always returns the elbow-up solution (q3 > 0).
    """
    # 1. base yaw and horizontal distance
    q1 = math.atan2(y, x)
    r = math.hypot(x, y)

    # 2. wrist point: step back from the target along the gripper direction
    rw = r - D * math.sin(phi3)
    zw = z - D * math.cos(phi3) - H        # height above the shoulder axis

    # 3. elbow angle from the law of cosines
    Dc = (rw**2 + zw**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(Dc) > 1:
        return None                         # target too far or too close
    q3 = math.acos(Dc)                      # + root = elbow up

    # 4. shoulder angle
    alpha = math.atan2(rw, zw)              # direction to the wrist point
    beta = math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
    q2 = alpha - beta

    # 5. wrist pitch makes the total angle equal phi3
    q4 = phi3 - q2 - q3

    # 6. wrist roll: align fingers with the object (symmetric under pi)
    q5 = obj_angle - q1
    q5 = ((q5 + math.pi / 2) % math.pi) - math.pi / 2

    q = (q1, q2, q3, q4, q5)

    # 7. joint limits
    if any(abs(v) > LIM for v in q):
        return None

    # 8. elbow and wrist must stay above the table
    elbow, wrist, _ = fk_planar(q2, q3, q4)
    if elbow[1] <= 0 or wrist[1] <= 0:
        return None

    return q
