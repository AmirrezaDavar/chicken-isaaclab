# SPDX-License-Identifier: BSD-3-Clause
"""Rotate humanoid base frame (+X forward alignment helper).

This script applies a yaw rotation to the base link frame and compensates all
joint anchors attached to that base link (localPos/localRot on the base side)
with the inverse rotation.

Use this when the robot's visual "forward" direction is not aligned with the
control convention (base +X).

Example:
    ./isaaclab.sh -p my_scripts/fix_humanoid_base_frame.py --headless \
        --input-usd my_assets/humanoid_moveable_default_params.usd \
        --output-usd my_assets/humanoid_moveable_default_params_base_x.usd \
        --robot-root /humanoid --base-link base_link --yaw-deg -90
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Fix humanoid base frame orientation in USD.")
parser.add_argument(
    "--input-usd",
    type=str,
    default="my_assets/humanoid_moveable_default_params.usd",
    help="Source USD path.",
)
parser.add_argument(
    "--output-usd",
    type=str,
    default="my_assets/humanoid_moveable_default_params_base_x.usd",
    help="Output USD path.",
)
parser.add_argument(
    "--robot-root",
    type=str,
    default="/humanoid",
    help="Robot root prim path. Example: /humanoid",
)
parser.add_argument(
    "--base-link",
    type=str,
    default="base_link",
    help="Base link name under --robot-root, or absolute path.",
)
parser.add_argument(
    "--yaw-deg",
    type=float,
    default=-90.0,
    help="Yaw rotation [deg] applied to base frame around +Z (local, post-multiply).",
)
parser.add_argument(
    "--dry-run",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Print planned changes without writing output USD.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim.utils.transforms as tf_utils


@dataclass
class FixSummary:
    base_prim_path: str
    yaw_deg: float
    joints_scanned: int = 0
    joints_updated: int = 0
    side0_updates: int = 0
    side1_updates: int = 0


def _to_abs_base_path(robot_root: str, base_link: str) -> str:
    if base_link.startswith("/"):
        return base_link
    return f"{robot_root.rstrip('/')}/{base_link}"


def _quatd_from_yaw_deg(yaw_deg: float) -> Gf.Quatd:
    half = math.radians(yaw_deg) * 0.5
    return Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))


def _quatd_from_any(q) -> Gf.Quatd:
    if q is None:
        return Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    if isinstance(q, Gf.Quatd):
        return q
    if isinstance(q, Gf.Quatf):
        i = q.GetImaginary()
        return Gf.Quatd(float(q.GetReal()), Gf.Vec3d(float(i[0]), float(i[1]), float(i[2])))
    i = q.GetImaginary()
    return Gf.Quatd(float(q.GetReal()), Gf.Vec3d(float(i[0]), float(i[1]), float(i[2])))


def _cast_quat_like(qd: Gf.Quatd, ref):
    i = qd.GetImaginary()
    if isinstance(ref, Gf.Quatf):
        return Gf.Quatf(float(qd.GetReal()), Gf.Vec3f(float(i[0]), float(i[1]), float(i[2])))
    return Gf.Quatd(float(qd.GetReal()), Gf.Vec3d(float(i[0]), float(i[1]), float(i[2])))


def _vec3d_from_any(v) -> Gf.Vec3d:
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


def _cast_vec3_like(vd: Gf.Vec3d, ref):
    if isinstance(ref, Gf.Vec3f):
        return Gf.Vec3f(float(vd[0]), float(vd[1]), float(vd[2]))
    return Gf.Vec3d(float(vd[0]), float(vd[1]), float(vd[2]))


def _rotate_vec_by_quat(v, q: Gf.Quatd) -> Gf.Vec3d:
    return Gf.Rotation(q).TransformDir(_vec3d_from_any(v))


def _is_joint_prim(prim: Usd.Prim) -> bool:
    if not prim.IsValid():
        return False
    if prim.IsA(UsdPhysics.Joint):
        return True
    return prim.HasAttribute("physics:body0") or prim.HasAttribute("physics:body1")


def _fix_joint_side(joint: UsdPhysics.Joint, side: int, q_inv: Gf.Quatd) -> bool:
    if side == 0:
        pos_attr = joint.GetLocalPos0Attr()
        rot_attr = joint.GetLocalRot0Attr()
    else:
        pos_attr = joint.GetLocalPos1Attr()
        rot_attr = joint.GetLocalRot1Attr()

    updated = False
    old_pos = pos_attr.Get() if pos_attr else None
    if old_pos is not None:
        new_pos = _rotate_vec_by_quat(old_pos, q_inv)
        pos_attr.Set(_cast_vec3_like(new_pos, old_pos))
        updated = True

    old_rot = rot_attr.Get() if rot_attr else None
    if old_rot is not None:
        old_rot_d = _quatd_from_any(old_rot)
        new_rot_d = q_inv * old_rot_d
        rot_attr.Set(_cast_quat_like(new_rot_d, old_rot))
        updated = True

    return updated


def _rotate_base_link_local_frame(base_prim: Usd.Prim, q_fix: Gf.Quatd):
    xformable = UsdGeom.Xformable(base_prim)
    tf = Gf.Transform(xformable.GetLocalTransformation())
    old_t = Gf.Vec3d(tf.GetTranslation())
    old_q = Gf.Quatd(tf.GetRotation().GetQuat())
    old_s = Gf.Vec3d(tf.GetScale())
    new_q = old_q * q_fix

    tf_utils.standardize_xform_ops(
        base_prim,
        translation=(float(old_t[0]), float(old_t[1]), float(old_t[2])),
        orientation=(
            float(new_q.GetReal()),
            float(new_q.GetImaginary()[0]),
            float(new_q.GetImaginary()[1]),
            float(new_q.GetImaginary()[2]),
        ),
        scale=(float(old_s[0]), float(old_s[1]), float(old_s[2])),
    )


def apply_fix(stage: Usd.Stage, robot_root: str, base_path: str, yaw_deg: float, dry_run: bool) -> FixSummary:
    root_prim = stage.GetPrimAtPath(robot_root)
    if not root_prim.IsValid():
        raise RuntimeError(f"Invalid robot root prim: {robot_root}")

    base_prim = stage.GetPrimAtPath(base_path)
    if not base_prim.IsValid():
        raise RuntimeError(f"Invalid base prim: {base_path}")
    if not base_prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(f"Base prim is not Xformable: {base_path}")

    q_fix = _quatd_from_yaw_deg(yaw_deg)
    q_inv = q_fix.GetInverse()
    summary = FixSummary(base_prim_path=base_path, yaw_deg=yaw_deg)

    if not dry_run:
        _rotate_base_link_local_frame(base_prim, q_fix)

    for prim in Usd.PrimRange(root_prim):
        if not _is_joint_prim(prim):
            continue
        summary.joints_scanned += 1
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue

        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        body0_hit = any(str(p) == base_path for p in body0_targets)
        body1_hit = any(str(p) == base_path for p in body1_targets)
        if not body0_hit and not body1_hit:
            continue

        updated_this_joint = False
        if body0_hit:
            if dry_run:
                updated_this_joint = True
                summary.side0_updates += 1
            else:
                updated = _fix_joint_side(joint, side=0, q_inv=q_inv)
                if updated:
                    updated_this_joint = True
                    summary.side0_updates += 1
        if body1_hit:
            if dry_run:
                updated_this_joint = True
                summary.side1_updates += 1
            else:
                updated = _fix_joint_side(joint, side=1, q_inv=q_inv)
                if updated:
                    updated_this_joint = True
                    summary.side1_updates += 1
        if updated_this_joint:
            summary.joints_updated += 1
            print(f"[FIX] joint={prim.GetPath()} body0_hit={body0_hit} body1_hit={body1_hit}")

    return summary


def main():
    input_usd = os.path.abspath(args_cli.input_usd)
    output_usd = os.path.abspath(args_cli.output_usd)
    base_path = _to_abs_base_path(args_cli.robot_root, args_cli.base_link)

    stage = Usd.Stage.Open(input_usd)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {input_usd}")

    print(f"[INFO] input_usd={input_usd}")
    print(f"[INFO] output_usd={output_usd}")
    print(f"[INFO] robot_root={args_cli.robot_root}")
    print(f"[INFO] base_path={base_path}")
    print(f"[INFO] yaw_deg={args_cli.yaw_deg:+.2f}")
    print(
        "[INFO] Operation: rotate base local frame by yaw, then inverse-rotate "
        "joint anchors (localPos/localRot) on the base side."
    )

    summary = apply_fix(
        stage=stage,
        robot_root=args_cli.robot_root,
        base_path=base_path,
        yaw_deg=args_cli.yaw_deg,
        dry_run=args_cli.dry_run,
    )

    if args_cli.dry_run:
        print("[INFO] dry-run mode: no USD written.")
    else:
        os.makedirs(os.path.dirname(output_usd), exist_ok=True)
        if not stage.GetRootLayer().Export(output_usd):
            raise RuntimeError(f"Failed to export USD to: {output_usd}")
        print(f"[INFO] exported: {output_usd}")

    print(
        "[SUMMARY] "
        f"base={summary.base_prim_path} yaw_deg={summary.yaw_deg:+.2f} "
        f"joints_scanned={summary.joints_scanned} joints_updated={summary.joints_updated} "
        f"side0_updates={summary.side0_updates} side1_updates={summary.side1_updates}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
