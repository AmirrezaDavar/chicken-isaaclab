from __future__ import annotations

import argparse
import builtins
import csv
import importlib
import math
import os
import re
from collections.abc import Sequence
from typing import Any
from xml.etree import ElementTree as ET

# If we're not already inside a launched Isaac Sim app, bootstrap a headless app.
SIMULATION_APP = None
if not hasattr(builtins, "ISAACSIM_APP_LAUNCHED"):
    from isaacsim import SimulationApp

    SIMULATION_APP = SimulationApp({"headless": True})

from pxr import PhysxSchema, Usd, UsdPhysics


DEFAULT_OUT_DIR = os.path.expanduser("~/isaac_param_dump")
JOINT_TYPE_NAMES = {"PhysicsJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
PRUNE_TYPE_NAMES = {
    "Mesh",
    "BasisCurves",
    "Points",
    "PointInstancer",
    "GeomSubset",
    "Material",
    "Shader",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump default USD/PhysX robot parameters to CSV.")
    parser.add_argument(
        "--usd-path",
        type=str,
        default="",
        help="USD file path to inspect directly (recommended for default USD parameters).",
    )
    parser.add_argument(
        "--robot-root",
        type=str,
        default="",
        help="Root prim path of the robot (e.g. /World/Humanoid). If empty, auto-detect.",
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for CSV files.")
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="",
        help="Optional URDF file path to dump URDF joint/link/inertia parameters and compare against USD dump.",
    )
    parser.add_argument(
        "--asset-cfg",
        type=str,
        default="",
        help="Optional Isaac Lab asset cfg reference module:object (e.g. isaaclab_assets:CLASS_HUMANOID_CFG).",
    )
    parser.add_argument("--summary-lines", type=int, default=8, help="How many lines to print from each CSV file.")
    parser.add_argument(
        "--disable-prune-geometry",
        action="store_true",
        help="Disable geometry/material subtree pruning while scanning prims (slower).",
    )
    args, _ = parser.parse_known_args()
    return args


def safe_get_attr(api_obj: Any, attr_name: str):
    """Return attribute value if getter exists and attr is valid."""
    if api_obj is None:
        return None
    getter = getattr(api_obj, f"Get{attr_name}Attr", None)
    if getter is None:
        return None
    try:
        attr = getter()
        if not attr:
            return None
        return attr.Get()
    except Exception:
        return None


def prim_has_api(prim: Usd.Prim, api_cls) -> bool:
    try:
        return prim.HasAPI(api_cls)
    except Exception:
        return False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv(rows: list[dict[str, Any]], out_csv: str, default_fieldnames: list[str]):
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else default_fieldnames
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        return float(value)
    except Exception:
        return None


def safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def read_csv_rows(csv_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def load_stage(usd_path: str) -> Usd.Stage:
    if usd_path:
        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            raise RuntimeError(f"Failed to open USD file: {usd_path}")
        print(f"[INFO] Loaded USD file: {usd_path}")
        return stage

    try:
        import omni
    except Exception as exc:
        raise RuntimeError(
            "--usd-path is not set, and omni.usd stage is unavailable. "
            "Use --usd-path to inspect a USD file directly."
        ) from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No live USD stage is open.")
    print("[INFO] Using current live USD stage.")
    return stage


def infer_robot_root_prim(stage: Usd.Stage) -> Usd.Prim:
    def to_robot_container(prim: Usd.Prim) -> Usd.Prim:
        cur = prim
        while cur.IsValid():
            parent = cur.GetParent()
            parent_path = parent.GetPath().pathString if parent and parent.IsValid() else ""
            if parent_path in {"/", "/World"}:
                return cur
            if not parent or not parent.IsValid():
                return cur
            cur = parent
        return prim

    # 1) Prefer articulation root prims.
    art_candidates: list[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if prim_has_api(prim, UsdPhysics.ArticulationRootAPI) or prim_has_api(prim, PhysxSchema.PhysxArticulationAPI):
            art_candidates.append(to_robot_container(prim))
    if art_candidates:
        unique_by_path = {p.GetPath().pathString: p for p in art_candidates}
        chosen = sorted(unique_by_path.values(), key=lambda p: len(p.GetPath().pathString))[0]
        print(f"[INFO] Auto-detected robot root from articulation API: {chosen.GetPath()}")
        return chosen

    # 2) Fallback from first joint by climbing up to direct child under /World or /.
    for prim in stage.Traverse():
        if not prim.IsValid() or prim.GetTypeName() not in JOINT_TYPE_NAMES:
            continue
        cur = prim.GetParent()
        while cur.IsValid():
            parent = cur.GetParent()
            parent_path = parent.GetPath().pathString if parent and parent.IsValid() else ""
            if parent_path in {"/", "/World"}:
                print(f"[INFO] Auto-detected robot root from joint ancestry: {cur.GetPath()}")
                return cur
            cur = parent
        break

    # 3) Stage default prim, if any.
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        print(f"[INFO] Auto-detected robot root from defaultPrim: {default_prim.GetPath()}")
        return default_prim

    # 4) /World fallback.
    world = stage.GetPrimAtPath("/World")
    if world and world.IsValid():
        print("[WARN] Falling back to /World as robot root.")
        return world

    # 5) Pseudo-root fallback.
    print("[WARN] Falling back to pseudo-root '/' as robot root.")
    return stage.GetPseudoRoot()


def get_root_prim(stage: Usd.Stage, robot_root: str) -> Usd.Prim:
    if robot_root:
        root_prim = stage.GetPrimAtPath(robot_root)
        if not root_prim.IsValid():
            raise ValueError(f"Invalid --robot-root: {robot_root}")
        print(f"[INFO] Using robot root: {robot_root}")
        return root_prim
    return infer_robot_root_prim(stage)


def iter_robot_prims(root_prim: Usd.Prim, prune_geometry: bool):
    """Yield prims under root. Optionally prune heavy geometry/material subtrees."""
    prim_range = Usd.PrimRange(root_prim)
    iterator = iter(prim_range)
    for prim in iterator:
        if not prim.IsValid():
            continue
        if prune_geometry and prim.GetTypeName() in PRUNE_TYPE_NAMES:
            iterator.PruneChildren()
        yield prim


def collect_joint_rows(root_prim: Usd.Prim, prune_geometry: bool = True) -> list[dict[str, Any]]:
    rows = []
    for prim in iter_robot_prims(root_prim, prune_geometry=prune_geometry):
        has_joint_attrs = prim.HasAttribute("physics:body0") or prim.HasAttribute("physics:body1")
        has_physx_joint_api = prim_has_api(prim, PhysxSchema.PhysxJointAPI)
        is_typed_joint = prim.GetTypeName() in JOINT_TYPE_NAMES
        if not (is_typed_joint or has_joint_attrs or has_physx_joint_api):
            continue

        joint_api = UsdPhysics.Joint(prim)
        physx_joint_api = PhysxSchema.PhysxJointAPI(prim) if prim_has_api(prim, PhysxSchema.PhysxJointAPI) else None
        drive_angular = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive_linear = UsdPhysics.DriveAPI.Get(prim, "linear")

        row = {
            "prim_path": str(prim.GetPath()),
            "name": prim.GetName(),
            "joint_type": prim.GetTypeName(),
            "body0": safe_get_attr(joint_api, "Body0"),
            "body1": safe_get_attr(joint_api, "Body1"),
            "local_pos0": safe_get_attr(joint_api, "LocalPos0"),
            "local_pos1": safe_get_attr(joint_api, "LocalPos1"),
            "local_rot0": safe_get_attr(joint_api, "LocalRot0"),
            "local_rot1": safe_get_attr(joint_api, "LocalRot1"),
            "break_force": safe_get_attr(joint_api, "BreakForce"),
            "break_torque": safe_get_attr(joint_api, "BreakTorque"),
            "joint_enabled": safe_get_attr(joint_api, "JointEnabled"),
            "ang_drive_type": safe_get_attr(drive_angular, "Type") if drive_angular else None,
            "ang_target_position": safe_get_attr(drive_angular, "TargetPosition") if drive_angular else None,
            "ang_target_velocity": safe_get_attr(drive_angular, "TargetVelocity") if drive_angular else None,
            "ang_stiffness": safe_get_attr(drive_angular, "Stiffness") if drive_angular else None,
            "ang_damping": safe_get_attr(drive_angular, "Damping") if drive_angular else None,
            "ang_max_force": safe_get_attr(drive_angular, "MaxForce") if drive_angular else None,
            "lin_drive_type": safe_get_attr(drive_linear, "Type") if drive_linear else None,
            "lin_target_position": safe_get_attr(drive_linear, "TargetPosition") if drive_linear else None,
            "lin_target_velocity": safe_get_attr(drive_linear, "TargetVelocity") if drive_linear else None,
            "lin_stiffness": safe_get_attr(drive_linear, "Stiffness") if drive_linear else None,
            "lin_damping": safe_get_attr(drive_linear, "Damping") if drive_linear else None,
            "lin_max_force": safe_get_attr(drive_linear, "MaxForce") if drive_linear else None,
            "physx_joint_friction": safe_get_attr(physx_joint_api, "JointFriction") if physx_joint_api else None,
            "physx_max_joint_velocity": safe_get_attr(physx_joint_api, "MaxJointVelocity") if physx_joint_api else None,
        }

        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute = UsdPhysics.RevoluteJoint(prim)
            row["axis"] = safe_get_attr(revolute, "Axis")
            row["lower_limit"] = safe_get_attr(revolute, "LowerLimit")
            row["upper_limit"] = safe_get_attr(revolute, "UpperLimit")
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic = UsdPhysics.PrismaticJoint(prim)
            row["axis"] = safe_get_attr(prismatic, "Axis")
            row["lower_limit"] = safe_get_attr(prismatic, "LowerLimit")
            row["upper_limit"] = safe_get_attr(prismatic, "UpperLimit")

        rows.append(row)
    return rows


def collect_rigid_body_rows(root_prim: Usd.Prim, prune_geometry: bool = True) -> list[dict[str, Any]]:
    rows = []
    for prim in iter_robot_prims(root_prim, prune_geometry=prune_geometry):
        # RigidBodyAPI is an API schema, so HasAPI should be used.
        if not prim_has_api(prim, UsdPhysics.RigidBodyAPI):
            continue

        rb_api = UsdPhysics.RigidBodyAPI(prim)
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI(prim) if prim_has_api(prim, PhysxSchema.PhysxRigidBodyAPI) else None
        mass_api = UsdPhysics.MassAPI(prim) if prim_has_api(prim, UsdPhysics.MassAPI) else None

        rows.append(
            {
                "prim_path": str(prim.GetPath()),
                "name": prim.GetName(),
                "rigid_body_enabled": safe_get_attr(rb_api, "RigidBodyEnabled"),
                "kinematic_enabled": safe_get_attr(rb_api, "KinematicEnabled"),
                "starts_asleep": safe_get_attr(rb_api, "StartsAsleep"),
                "mass": safe_get_attr(mass_api, "Mass") if mass_api else None,
                "density": safe_get_attr(mass_api, "Density") if mass_api else None,
                "center_of_mass": safe_get_attr(mass_api, "CenterOfMass") if mass_api else None,
                "diagonal_inertia": safe_get_attr(mass_api, "DiagonalInertia") if mass_api else None,
                "principal_axes": safe_get_attr(mass_api, "PrincipalAxes") if mass_api else None,
                "physx_disable_gravity": safe_get_attr(physx_rb_api, "DisableGravity") if physx_rb_api else None,
                "physx_linear_damping": safe_get_attr(physx_rb_api, "LinearDamping") if physx_rb_api else None,
                "physx_angular_damping": safe_get_attr(physx_rb_api, "AngularDamping") if physx_rb_api else None,
                "physx_max_linear_velocity": safe_get_attr(physx_rb_api, "MaxLinearVelocity") if physx_rb_api else None,
                "physx_max_angular_velocity": safe_get_attr(physx_rb_api, "MaxAngularVelocity") if physx_rb_api else None,
                "physx_max_depenetration_velocity": safe_get_attr(physx_rb_api, "MaxDepenetrationVelocity")
                if physx_rb_api
                else None,
                "physx_retain_accelerations": safe_get_attr(physx_rb_api, "RetainAccelerations") if physx_rb_api else None,
            }
        )
    return rows


def collect_articulation_rows(root_prim: Usd.Prim, prune_geometry: bool = True) -> list[dict[str, Any]]:
    rows = []
    for prim in iter_robot_prims(root_prim, prune_geometry=prune_geometry):
        has_articulation = prim_has_api(prim, UsdPhysics.ArticulationRootAPI)
        has_physx_articulation = prim_has_api(prim, PhysxSchema.PhysxArticulationAPI)
        if not (has_articulation or has_physx_articulation):
            continue

        art_api = UsdPhysics.ArticulationRootAPI(prim) if has_articulation else None
        physx_art_api = PhysxSchema.PhysxArticulationAPI(prim) if has_physx_articulation else None
        rows.append(
            {
                "prim_path": str(prim.GetPath()),
                "name": prim.GetName(),
                "articulation_enabled": safe_get_attr(art_api, "ArticulationEnabled") if art_api else None,
                "enabled_self_collisions": safe_get_attr(physx_art_api, "EnabledSelfCollisions") if physx_art_api else None,
                "solver_position_iteration_count": safe_get_attr(physx_art_api, "SolverPositionIterationCount")
                if physx_art_api
                else None,
                "solver_velocity_iteration_count": safe_get_attr(physx_art_api, "SolverVelocityIterationCount")
                if physx_art_api
                else None,
                "sleep_threshold": safe_get_attr(physx_art_api, "SleepThreshold") if physx_art_api else None,
                "stabilization_threshold": safe_get_attr(physx_art_api, "StabilizationThreshold") if physx_art_api else None,
            }
        )
    return rows


def dump_joint_focus_table(joint_rows: list[dict[str, Any]], out_csv: str):
    focus_rows = []
    for row in joint_rows:
        focus_rows.append(
            {
                "joint_name": row.get("name"),
                "joint_path": row.get("prim_path"),
                "joint_type": row.get("joint_type"),
                "axis": row.get("axis"),
                "lower_limit": row.get("lower_limit"),
                "upper_limit": row.get("upper_limit"),
                "stiffness": row.get("ang_stiffness"),
                "damping": row.get("ang_damping"),
                "effort_limit": row.get("ang_max_force"),
                "max_joint_velocity": row.get("physx_max_joint_velocity"),
                "joint_friction": row.get("physx_joint_friction"),
            }
        )
    write_csv(
        focus_rows,
        out_csv,
        default_fieldnames=[
            "joint_name",
            "joint_path",
            "joint_type",
            "axis",
            "lower_limit",
            "upper_limit",
            "stiffness",
            "damping",
            "effort_limit",
            "max_joint_velocity",
            "joint_friction",
        ],
    )
    print(f"[OK] Joint focus table saved: {out_csv}")


def dump_mass_summary(rigid_rows: list[dict[str, Any]], out_csv: str):
    numeric_masses = []
    for row in rigid_rows:
        mass = row.get("mass")
        try:
            if mass is not None:
                numeric_masses.append(float(mass))
        except Exception:
            continue

    summary_row = {
        "rigid_body_count": len(rigid_rows),
        "bodies_with_authored_mass": len(numeric_masses),
        "total_mass": sum(numeric_masses) if numeric_masses else None,
        "mean_mass": (sum(numeric_masses) / len(numeric_masses)) if numeric_masses else None,
    }
    write_csv([summary_row], out_csv, default_fieldnames=list(summary_row.keys()))
    print(f"[OK] Mass summary saved: {out_csv}")


def resolve_cfg_value_for_joint(value_cfg: Any, joint_name: str):
    if isinstance(value_cfg, dict):
        matched = [value for expr, value in value_cfg.items() if re.fullmatch(expr, joint_name)]
        if not matched:
            return None
        return matched[0]
    return value_cfg


def load_object_from_ref(ref: str):
    if ":" in ref:
        module_name, obj_name = ref.split(":", maxsplit=1)
    else:
        module_name, obj_name = ref.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def joint_matches_any_expr(joint_name: str, exprs: Sequence[str]) -> bool:
    return any(re.fullmatch(expr, joint_name) for expr in exprs)


def dump_asset_cfg_actuator_table(joint_rows: list[dict[str, Any]], asset_cfg_ref: str, out_csv: str):
    if not asset_cfg_ref:
        return
    try:
        asset_cfg = load_object_from_ref(asset_cfg_ref)
    except Exception as exc:
        print(f"[WARN] Failed to load --asset-cfg '{asset_cfg_ref}': {exc}")
        return

    actuators = getattr(asset_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        print(f"[WARN] Asset cfg '{asset_cfg_ref}' has no 'actuators' dict. Skip actuator table.")
        return

    joint_names = [row["name"] for row in joint_rows]
    rows = []
    matched_joint_names: set[str] = set()
    for group_name, actuator_cfg in actuators.items():
        joint_exprs = getattr(actuator_cfg, "joint_names_expr", None)
        if not joint_exprs:
            continue
        for joint_name in joint_names:
            if not joint_matches_any_expr(joint_name, joint_exprs):
                continue
            matched_joint_names.add(joint_name)
            rows.append(
                {
                    "joint_name": joint_name,
                    "actuator_group": group_name,
                    "stiffness_cfg": resolve_cfg_value_for_joint(getattr(actuator_cfg, "stiffness", None), joint_name),
                    "damping_cfg": resolve_cfg_value_for_joint(getattr(actuator_cfg, "damping", None), joint_name),
                    "effort_limit_sim_cfg": resolve_cfg_value_for_joint(
                        getattr(actuator_cfg, "effort_limit_sim", None), joint_name
                    ),
                    "effort_limit_cfg": resolve_cfg_value_for_joint(getattr(actuator_cfg, "effort_limit", None), joint_name),
                    "velocity_limit_sim_cfg": resolve_cfg_value_for_joint(
                        getattr(actuator_cfg, "velocity_limit_sim", None), joint_name
                    ),
                    "velocity_limit_cfg": resolve_cfg_value_for_joint(
                        getattr(actuator_cfg, "velocity_limit", None), joint_name
                    ),
                    "friction_cfg": resolve_cfg_value_for_joint(getattr(actuator_cfg, "friction", None), joint_name),
                    "armature_cfg": resolve_cfg_value_for_joint(getattr(actuator_cfg, "armature", None), joint_name),
                    "saturation_effort_cfg": resolve_cfg_value_for_joint(
                        getattr(actuator_cfg, "saturation_effort", None), joint_name
                    ),
                }
            )

    for joint_name in joint_names:
        if joint_name in matched_joint_names:
            continue
        rows.append(
            {
                "joint_name": joint_name,
                "actuator_group": "",
                "stiffness_cfg": None,
                "damping_cfg": None,
                "effort_limit_sim_cfg": None,
                "effort_limit_cfg": None,
                "velocity_limit_sim_cfg": None,
                "velocity_limit_cfg": None,
                "friction_cfg": None,
                "armature_cfg": None,
                "saturation_effort_cfg": None,
            }
        )

    write_csv(
        rows,
        out_csv,
        default_fieldnames=[
            "joint_name",
            "actuator_group",
            "stiffness_cfg",
            "damping_cfg",
            "effort_limit_sim_cfg",
            "effort_limit_cfg",
            "velocity_limit_sim_cfg",
            "velocity_limit_cfg",
            "friction_cfg",
            "armature_cfg",
            "saturation_effort_cfg",
        ],
    )
    print(f"[OK] Asset actuator table saved: {out_csv}")


def print_short_summary(csv_path: str, max_lines: int = 8):
    print(f"\n[SUMMARY] {csv_path}")
    try:
        with open(csv_path, "r", newline="") as f:
            for i, line in enumerate(f):
                print(line.rstrip())
                if i >= max_lines:
                    print("...")
                    break
    except Exception as exc:
        print(f"[WARN] Could not read summary: {exc}")


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    stage = load_stage(args.usd_path)
    root_prim = get_root_prim(stage, args.robot_root)

    prune_geometry = not args.disable_prune_geometry
    joints_csv = os.path.join(args.out_dir, "joints.csv")
    joints_focus_csv = os.path.join(args.out_dir, "joint_focus.csv")
    rigid_csv = os.path.join(args.out_dir, "rigid_bodies.csv")
    mass_summary_csv = os.path.join(args.out_dir, "mass_summary.csv")
    art_csv = os.path.join(args.out_dir, "articulation_roots.csv")
    actuator_cfg_csv = os.path.join(args.out_dir, "actuator_cfg.csv")

    joint_rows = collect_joint_rows(root_prim, prune_geometry=prune_geometry)
    rigid_rows = collect_rigid_body_rows(root_prim, prune_geometry=prune_geometry)
    art_rows = collect_articulation_rows(root_prim, prune_geometry=prune_geometry)

    write_csv(joint_rows, joints_csv, default_fieldnames=["prim_path", "name", "joint_type"])
    write_csv(rigid_rows, rigid_csv, default_fieldnames=["prim_path", "name"])
    write_csv(art_rows, art_csv, default_fieldnames=["prim_path", "name"])
    dump_joint_focus_table(joint_rows, joints_focus_csv)
    dump_mass_summary(rigid_rows, mass_summary_csv)
    dump_asset_cfg_actuator_table(joint_rows, args.asset_cfg, actuator_cfg_csv)

    print(f"[OK] Joint params saved: {joints_csv}")
    print(f"[INFO] #joints = {len(joint_rows)}")
    print(f"[OK] Rigid body params saved: {rigid_csv}")
    print(f"[INFO] #rigid bodies = {len(rigid_rows)}")
    print(f"[OK] Articulation params saved: {art_csv}")
    print(f"[INFO] #articulation roots = {len(art_rows)}")

    print_short_summary(joints_focus_csv, max_lines=args.summary_lines)
    print_short_summary(rigid_csv, max_lines=args.summary_lines)
    print_short_summary(mass_summary_csv, max_lines=args.summary_lines)
    print_short_summary(art_csv, max_lines=args.summary_lines)
    if args.asset_cfg:
        print_short_summary(actuator_cfg_csv, max_lines=args.summary_lines)

    print("\n[DONE]")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if SIMULATION_APP is not None:
            SIMULATION_APP.close()
