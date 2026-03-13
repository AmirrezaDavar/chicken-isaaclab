import argparse
import math
import xml.etree.ElementTree as ET
import numpy as np


def rpy_to_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])

    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])

    return Rz @ Ry @ Rx


def matrix_to_rpy(R):
    sy = -R[2,0]
    cy = math.sqrt(R[0,0]**2 + R[1,0]**2)

    if cy > 1e-6:
        r = math.atan2(R[2,1], R[2,2])
        p = math.atan2(sy, cy)
        y = math.atan2(R[1,0], R[0,0])
    else:
        r = math.atan2(-R[1,2], R[1,1])
        p = math.atan2(sy, cy)
        y = 0

    return r, p, y


def parse_xyz(x):
    return np.array([float(v) for v in x.split()])


def parse_rpy(x):
    return [float(v) for v in x.split()]


def format_vec(v):
    return f"{v[0]} {v[1]} {v[2]}"


def rotate_origin(origin_elem, R):
    xyz = origin_elem.get("xyz", "0 0 0")
    rpy = origin_elem.get("rpy", "0 0 0")

    xyz = parse_xyz(xyz)
    r, p, y = parse_rpy(rpy)

    R_local = rpy_to_matrix(r, p, y)

    xyz_new = R @ xyz
    R_new = R @ R_local

    r2, p2, y2 = matrix_to_rpy(R_new)

    origin_elem.set("xyz", format_vec(xyz_new))
    origin_elem.set("rpy", format_vec([r2, p2, y2]))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root-link", default="base_link")
    parser.add_argument("--yaw-deg", type=float, default=90)

    args = parser.parse_args()

    yaw = math.radians(args.yaw_deg)
    R = rpy_to_matrix(0, 0, yaw)

    tree = ET.parse(args.input)
    root = tree.getroot()

    # rotate base_link visuals / collisions / inertial
    for link in root.findall("link"):
        if link.get("name") == args.root_link:

            for tag in ["visual", "collision", "inertial"]:
                for elem in link.findall(tag):
                    origin = elem.find("origin")
                    if origin is not None:
                        rotate_origin(origin, R)

    # rotate joints attached to root
    for joint in root.findall("joint"):

        parent = joint.find("parent")
        if parent is None:
            continue

        if parent.get("link") == args.root_link:

            origin = joint.find("origin")
            if origin is not None:
                rotate_origin(origin, R)

    tree.write(args.output)


if __name__ == "__main__":
    main()
