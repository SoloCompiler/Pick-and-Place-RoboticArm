import math
import sys

from arm_math import (
    HOVER_Z, GRASP_Z, DROP_Z, CUBE_SIZE, MAX_GRASP_ERROR, MIN_SPACING, GRIPPER_OPEN,
    V_JOINT, V_APPROACH, V_TRANSIT, MIN_T_JOINT, MIN_T_LINE, MIN_T_TRANSIT,
    CARRY_WARN_Z, CARRY_WARN_DIST, JOINT_LIMITS, DEMO_JOBS, COLORS,
    joint_path, line_path, transit_path, plan_job, ik_arm, closest_to_path,
    reach_band, spacing_ok, drop_slots,
)
from arm_backends import (
    SimArm, SimDetector, ServoArm, DryRunWriter, SerialPulseWriter, HintDetector,
    HW_CALIBRATED, servo_test,
)


class PickPlaceController:
    def __init__(self, arm, detector, limits):
        self.arm, self.det, self.limits = arm, detector, limits

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

    def run_job(self, n, job, others=()):
        arm = self.arm
        print(f"\n--- Job {n} ---")
        hint, tgt = job["object"][:2], job["target"]

        obj = self.det.locate(hint, 0.06)
        if obj is None:
            print("SKIPPED: no object found near the given position")
            return "SKIPPED", "object not detected"
        ox, oy, oyaw = obj
        print(f"Detected object at ({100*ox:.1f}, {100*oy:.1f}) cm, yaw {math.degrees(oyaw):.0f} deg; "
              f"target ({100*tgt[0]:.1f}, {100*tgt[1]:.1f}) cm")

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
            others = [j["object"][:2] for j in jobs[i + 1:]] + [j["target"] for j in jobs[:i]]
            summary.append((i + 1, *self.run_job(i + 1, job, others)))
        print("\n=========== SUMMARY ===========")
        for n, status, info in summary:
            print(f"Job {n}: {status}  ({info})")
        return summary


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
                             "  first, paste its output into arm_backends.py and set HW_CALIBRATED = True.\n"
                             "(Wrong servo directions or zeros can drive the arm into the table.)")
        detector = load_camera()
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