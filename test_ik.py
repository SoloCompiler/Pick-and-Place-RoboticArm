"""Math-only tests for IK: worked example, round trip, workspace sweep."""
import math
import random
from kinematics import fk_3d, ik

# A) worked example from the handoff
q = ik(0.15, 0.0, 0.01, math.pi)
print("A) worked example:", None if q is None else [round(v, 3) for v in q])
print("   expect q2~0.535, q3~1.295, q4~1.312")

# B) round trip: random angles -> FK -> IK -> FK. Positions must match.
fails = accepted = 0
for _ in range(5000):
    q1 = random.uniform(-1.4, 1.4)
    q2, q3, q4 = (random.uniform(-1.4, 1.4) for _ in range(3))
    t = fk_3d(q1, q2, q3, q4)
    sol = ik(*t, phi3=q2 + q3 + q4)
    if sol is None:
        continue  # legitimately rejected (limits or table)
    accepted += 1
    b = fk_3d(*sol[:4])
    if max(abs(u - v) for u, v in zip(b, t)) > 1e-9:
        fails += 1
print(f"B) round trip: {accepted} accepted, {fails} mismatches (want 0)")

# C) workspace sweep at grasp height 1 cm
print("C) reachable table area at z = 0.01 m")
for phi3 in (math.pi, 2.8, 2.6, 2.4):
    ok = [(x / 100, y / 100) for x in range(-25, 26) for y in range(-25, 26)
          if ik(x / 100, y / 100, 0.01, phi3) is not None]
    if ok:
        rs = [math.hypot(*pt) for pt in ok]
        print(f"   phi3={phi3:.2f}: {len(ok)} points, radius {min(rs):.3f} to {max(rs):.3f} m")
    else:
        print(f"   phi3={phi3:.2f}: none")
