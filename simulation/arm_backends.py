import math
import time
import numpy as np
 
from arm_math import (URDF_PATH, JOINT_LIMITS, GRIPPER_OPEN, GRIPPER_MIN_GAP, CUBE_SIZE,
                      GRASP_ASSIST, REALTIME, SPEED, SIM_DT, COLORS)
 
 
class ArmInterface:
    pass
 
 
class DetectorInterface:
    pass
 
 
class SimArm(ArmInterface):
    dt = SIM_DT
    grasp_close_opening = 0.0
 
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
        self.gripper_link = self.idx["j5_wrist_roll"]
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
        for i in (self.grip, self.follow):
            self.p.setJointMotorControl2(self.body, i, self.p.POSITION_CONTROL, targetPosition=opening, force=20)
 
    def holding_object(self, width):
        f1, f2 = (self.p.getJointState(self.body, i)[0] for i in (self.grip, self.follow))
        expected = width - GRIPPER_MIN_GAP
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
        # createConstraint measures parentFramePosition from the link's CENTRE-OF-MASS frame,
        # not the URDF link frame. [0:2] = COM frame, [4:6] = URDF link frame. Using [4:6] here
        # shifted the cube by the gripper's inertial offset (22.5 mm) along the approach axis.
        lpos, lorn = p.getLinkState(self.body, self.gripper_link)[0:2]
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
 
 
HW_CALIBRATED = False
 
US_PER_RAD = 1400.0 / math.radians(165.0)
SERVO_CFG = [
    dict(name="j1_base",        ch=0, sign=+1, zero_us=1500),
    dict(name="j2_shoulder",    ch=1, sign=+1, zero_us=1500),
    dict(name="j3_elbow",       ch=2, sign=+1, zero_us=1500),
    dict(name="j4_wrist_pitch", ch=3, sign=+1, zero_us=1500),
    dict(name="j5_wrist_roll",  ch=4, sign=+1, zero_us=1500),
]
GRIP_CH = 5
GRIP_OPEN_US, GRIP_CLOSED_US = 1700, 1300
US_MIN, US_MAX = 800, 2200
HW_DT = 0.02
HW_MAX_STEP = 1.5 * HW_DT
GRIP_STEP = 0.0015
 
JOINT_HINTS = {
    "j1_base":        "+ turns the whole arm COUNTER-clockwise seen from above (toward the LEFT, +Y)",
    "j2_shoulder":    "+ leans the whole arm FORWARD (away from the base, toward +X)",
    "j3_elbow":       "+ folds the forearm FORWARD relative to the upper arm",
    "j4_wrist_pitch": "+ tips the gripper FORWARD relative to the forearm",
    "j5_wrist_roll":  "+ rolls the gripper counter-clockwise when you look at it from the fingertip end",
}
 
 
class SerialPulseWriter:
    def __init__(self, port, baud=115200):
        try:
            import serial
        except ImportError:
            raise SystemExit("pyserial is missing:  pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)
 
    def __call__(self, pulses):
        self.ser.write(("S " + " ".join(str(int(round(u))) for u in pulses) + "\n").encode())
 
    def close(self):
        self.ser.close()
 
 
class DryRunWriter:
    def __init__(self):
        self.n, self.lo, self.hi, self.last, self.max_jump = 0, [10 ** 9] * 6, [0] * 6, None, 0.0
 
    def __call__(self, pulses):
        self.n += 1
        for i, u in enumerate(pulses):
            self.lo[i], self.hi[i] = min(self.lo[i], u), max(self.hi[i], u)
        if self.last is not None:
            self.max_jump = max(self.max_jump, max(abs(a - b) for a, b in zip(pulses[:5], self.last[:5])))
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
    grasp_close_opening = (CUBE_SIZE - GRIPPER_MIN_GAP) / 2 - 0.003
 
    def __init__(self, write_pulses, limits=JOINT_LIMITS, realtime=True):
        self.write, self.limits, self.realtime = write_pulses, limits, realtime
        self._t = 0.0
        self.q = [0.0] * 5
        self.opening = GRIPPER_OPEN
        self._send()
 
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
            qi = min(max(qi, lo), hi)
            qi = old + max(-HW_MAX_STEP, min(HW_MAX_STEP, qi - old))
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
        self.wait(0.3)
        return True
 
    def set_gripper(self, opening):
        target = min(max(opening, 0.0), GRIPPER_OPEN)
        while abs(self.opening - target) > 1e-9:
            self.opening += max(-GRIP_STEP, min(GRIP_STEP, target - self.opening))
            self._send()
            self._tick()
 
    def holding_object(self, width):
        return None
 
    def tcp_position(self):
        return None
 
    def on_grasp(self):
        pass
 
    def on_release(self):
        pass
 
    def torque_report(self, q):
        return ""
 
 
class CameraDetector(DetectorInterface):
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
            x2, y2 = self._to_table(u + 10 * math.cos(ang), v + 10 * math.sin(ang))
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
    def __init__(self, jobs):
        self.objs = [tuple(j["object"]) for j in jobs]
 
    def locate_all(self):
        return list(self.objs)
 
    def locate(self, hint_xy, radius=0.06):
        near = [o for o in self.objs if math.dist(o[:2], hint_xy) < radius]
        return min(near, key=lambda o: math.dist(o[:2], hint_xy)) if near else None
 
 
def servo_test(port):
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
        print("\n=== paste these lines into arm_backends.py ===")
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
        for i in range(1, 101):
            q = list(arm.q)
            q[k] = base + (target - base) * i / 100
            arm.command(q)
        arm.wait(0.5)
        ans = input("Did it move THAT way (as described above)? [y/n/r = repeat]: ").strip().lower()
        for i in range(1, 101):
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

