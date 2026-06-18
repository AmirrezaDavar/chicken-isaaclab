# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manipulation environments for fixed-arm robots."""

from .reach import *  # noqa
from .chicken_lift.config.ur10e import *  # noqa: registers Isaac-Lift-Chicken-UR10e-v0
from .chicken_lift.config.ur10e_custom_gripper import *  # noqa: registers Isaac-Lift-Chicken-UR10e-CustomGripper-v0
from .cube_lift.config.ur10e_custom_gripper import *  # noqa: registers Isaac-Lift-Cube-UR10e-CustomGripper-v0
