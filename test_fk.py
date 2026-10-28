"""Compare my FK with the TCP position that PyBullet computes from the URDF.
Run this first. If the error is large, fix FK before touching IK."""
import random
import pybullet as p
import pybullet_data
from kinematics import fk_3d

p.connect(p.DIRECT)  # no window
p.setAdditionalSearchPath(pybullet_data.getDataPath())
arm = p.loadURDF(r"C:\armproject\arm6dof_v2.urdf", useFixedBase=True)

J = {p.getJointInfo(arm, i)[1].decode(): i for i in range(p.getNumJoints(arm))}
tcp = J["tcp_fixed"]
names = ["j1_base", "j2_shoulder", "j3_elbow", "j4_wrist_pitch"]

worst = 0.0
for _ in range(50):
    q = [random.uniform(-1.2, 1.2) for _ in range(4)]
    for n, v in zip(names, q):
        p.resetJointState(arm, J[n], v)
    sim = p.getLinkState(arm, tcp, computeForwardKinematics=1)[4]
    mine = fk_3d(*q)
    worst = max(worst, max(abs(a - b) for a, b in zip(sim, mine)))

print("worst FK error vs URDF (m):", worst)
print("PASS" if worst < 0.001 else "FAIL - check signs/constants")
