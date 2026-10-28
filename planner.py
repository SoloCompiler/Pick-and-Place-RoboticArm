"""Automatic gripper-angle selection on top of kinematics.ik."""
import math
from kinematics import ik


def ik_auto(x, y, z, obj_angle=0.0, phi_min=2.0, step=0.05):
    """Try the straight-down grasp first, then tilt the gripper a little at a
    time until IK accepts. Returns (q1..q5, phi3) or None if nothing works.
    phi3 = pi is straight down; smaller values tilt the gripper outward."""
    phi3 = math.pi
    while phi3 >= phi_min:
        q = ik(x, y, z, phi3, obj_angle)
        if q is not None:
            return q, phi3
        phi3 -= step
    return None


if __name__ == "__main__":
    # Workspace report for the teammate: how far out can objects be placed?
    for z in (0.01, 0.03):
        rs = []
        for xi in range(-35, 36):
            for yi in range(-35, 36):
                x, y = xi / 100, yi / 100
                if ik_auto(x, y, z) is not None:
                    rs.append(math.hypot(x, y))
        print(f"z={z:.2f} m: {len(rs)} reachable points, "
              f"radius {min(rs):.3f} to {max(rs):.3f} m")
