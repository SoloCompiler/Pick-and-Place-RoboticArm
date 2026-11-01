"""Visual check: put a red marker at each target, move the arm there with IK,
and print how far the gripper tip is from the marker (should be ~0 mm)."""
import math
import os
import time
import pybullet as p
import pybullet_data
from planner import ik_auto

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm6dof_v2.urdf")
arm = p.loadURDF(urdf, useFixedBase=True)

J = {p.getJointInfo(arm, i)[1].decode(): i for i in range(p.getNumJoints(arm))}
arm_joints = ["j1_base", "j2_shoulder", "j3_elbow", "j4_wrist_pitch", "j5_wrist_roll"]
tcp = J["tcp_fixed"]

# open the gripper (follower copies the main finger, as in prototype.py)
p.resetJointState(arm, J["j6_gripper"], 0.02)
p.resetJointState(arm, J["j6_gripper_follower"], 0.02)

# (x, y, z, object angle) in metres / radians, base frame
targets = [
    (0.15, 0.00, 0.01, 0.0),
    (0.10, 0.10, 0.01, math.pi / 4),   # r = 0.141, straight down
    (0.15, -0.08, 0.01, 0.0),
    (0.20, 0.10, 0.01, 0.5),           # r = 0.224, needs a tilt
    (0.25, 0.00, 0.01, 0.0),           # needs more tilt
    (0.35, 0.00, 0.01, 0.0),           # out of reach: should be rejected
]

marker = vis = None
vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[1, 0, 0, 1])
p.resetDebugVisualizerCamera(0.6, 45, -30, [0.1, 0, 0.08])

for (x, y, z, ang) in targets:
    if marker is not None:
        p.removeBody(marker)
    marker = p.createMultiBody(0, -1, vis, basePosition=[x, y, z])

    res = ik_auto(x, y, z, ang)
    if res is None:
        print(f"target ({x:.2f}, {y:.2f}, {z:.2f}): UNREACHABLE")
        time.sleep(1.5)
        continue
    q, phi3 = res
    for name, v in zip(arm_joints, q):
        p.resetJointState(arm, J[name], v)
    tip = p.getLinkState(arm, tcp, computeForwardKinematics=1)[4]
    err_mm = 1000 * math.dist(tip, (x, y, z))
    print(f"target ({x:.2f}, {y:.2f}, {z:.2f}): phi3={phi3:.2f}, "
          f"tip error {err_mm:.3f} mm, q={[round(v, 3) for v in q]}")
    time.sleep(2.0)

print("done - close the window to exit")
while p.isConnected():
    time.sleep(0.1)
