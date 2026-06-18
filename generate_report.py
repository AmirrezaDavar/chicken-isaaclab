#!/usr/bin/env python3
"""Generate the final Word document report for CSCE50103 Project 2."""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PLOTS_DIR = "/home/wanglab22/CSCE50103-IsaacLab/logs/presentation_plots"

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h

def add_paragraph(doc, text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p

def add_figure(doc, filename, caption, width=5.5):
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)
    else:
        doc.add_paragraph(f"[Figure not found: {filename}]")

def add_table_row(table, cells, bold=False, shade=None):
    row = table.add_row()
    for i, text in enumerate(cells):
        cell = row.cells[i]
        cell.text = text
        if bold:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.bold = True
        if shade:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), shade)
            tcPr.append(shd)
    return row

def set_table_header(table, headers):
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = h
        for run in hdr.cells[i].paragraphs[0].runs:
            run.bold = True
        tc = hdr.cells[i]._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'D9D9D9')
        tcPr.append(shd)

doc = Document()

# ── Title page ──────────────────────────────────────────────────────────────
title = doc.add_heading('Humanoid Robot Reinforcement Learning', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph('CSE50103 — Spring 2026 | Final Project Report')
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(12)

authors = doc.add_paragraph(
    'Taisei Hanyu · Robert Russell · Himanshu · Amirreza Davar · David Nystrom'
)
authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
authors.runs[0].font.size = Pt(11)

doc.add_paragraph()

# ── Abstract ────────────────────────────────────────────────────────────────
add_heading(doc, '1. Abstract', 1)
doc.add_paragraph(
    'This report presents the design, training, and evaluation of reinforcement learning (RL) '
    'policies for a humanoid robot in NVIDIA Isaac Lab (Isaac Sim 5.1). '
    'We implement four primitive skills — reaching, stepping, squatting, and walking — '
    'and two high-level manipulation skills: pushing a wall-mounted button and shooting a '
    'floating ball into a goal sphere. '
    'All locomotion policies are trained with Proximal Policy Optimization (PPO) via RSL-RL '
    'across up to 1,536 parallel environments. '
    'Key contributions include a reward-shaping framework for stable bipedal manipulation, '
    'a depth-camera–based reaching pipeline using a TiledCamera and DepthTargetPosCommand, '
    'diagnosis and correction of three critical reward bugs in the shoot-ball task, and '
    'a multi-run fine-tuning strategy that brought reaching reward from −5.0 to near 0.'
)

# ── Project Overview ─────────────────────────────────────────────────────────
add_heading(doc, '2. Project Overview', 1)
doc.add_paragraph(
    'The project builds on a URDF → USD pipeline (Tasks A–D from the midterm milestone) '
    'to train manipulation-capable humanoid policies. '
    'The final milestone focuses on three deliverable sets:'
)
doc.add_paragraph('• Task E — Primitive Skills: reaching, stepping, squatting, walking', style='List Bullet')
doc.add_paragraph('• Task F — High-Level Skills: push a button, shoot a ball into a target', style='List Bullet')
doc.add_paragraph('• Task G — Isaac Lab RL: reward design, observation engineering, domain randomization, and depth-camera comparison', style='List Bullet')

doc.add_paragraph(
    'All RL training uses RSL-RL PPO. The Class Humanoid robot has 21 actuated joints '
    'controlled via position targets. The framework is NVIDIA Isaac Lab running on '
    'Isaac Sim 5.1 with PhysX GPU simulation.'
)

# ── Task E ──────────────────────────────────────────────────────────────────
add_heading(doc, '3. Task E: Primitive Skills', 1)
doc.add_paragraph(
    'The methodology for all primitive skills is iterative: assign training weights, '
    'run simulation, evaluate whether training is successful, and adjust reward weights '
    'or training setup if needed. All runs share a common baseline locomotion '
    'configuration (LocomotionVelocityRoughEnvCfg) extended per task.'
)

# ── 3.1 Reaching ────────────────────────────────────────────────────────────
add_heading(doc, '3.1 Right-Arm Reaching', 2)
doc.add_paragraph(
    'Goal: drive the right wrist within 5 cm of a randomly-placed target sphere '
    '(x ∈ [0.30, 0.50] m, y ∈ [−0.15, 0.0] m, z ∈ [0.60, 0.72] m in the robot base frame).'
)

add_heading(doc, 'Observation Space', 3)
obs_items = [
    'base_lin_vel (3)', 'base_ang_vel (3)', 'projected_gravity (3)',
    'joint_pos (all joints)', 'joint_vel (all joints)', 'actions (previous)',
    'target_pos_base_from_depth (3) — ground-truth or depth-camera-derived target position'
]
for item in obs_items:
    doc.add_paragraph(f'• {item}', style='List Bullet')

add_heading(doc, 'Reward Structure', 3)
t = doc.add_table(rows=1, cols=3)
t.style = 'Table Grid'
set_table_header(t, ['Reward Term', 'Weight', 'Description'])
reward_rows = [
    ('reach_target', '+8.0', 'tanh(−dist/0.08) — large positive bonus near target'),
    ('reach_distance', '−0.5', 'L2 distance: wrist to target'),
    ('arm_joint_deviation', '−0.02', 'L1 deviation of right-arm joints from default'),
    ('undesired_contacts', '−0.5', 'Penalizes contact on non-foot bodies (threshold 1 N)'),
]
for row in reward_rows:
    add_table_row(t, row)
doc.add_paragraph()

add_heading(doc, 'Training Runs', 3)
t2 = doc.add_table(rows=1, cols=4)
t2.style = 'Table Grid'
set_table_header(t2, ['Run', 'Iterations', 'Start', 'Final Reward'])
training_rows = [
    ('reach_v7 (baseline)', '2,600', 'Scratch', '≈ −2.8'),
    ('reach_v10 (fine-tune from v7)', '4,000', 'v7 checkpoint', '≈ 0.0 (best)'),
    ('reach_v11 (fine-tune from v10)', '2,350', 'v10 checkpoint', '≈ −4.0 (plateau)'),
    ('reach_left_fixed_v1 (left arm, fixed base)', '3,999', 'Scratch', 'Converged'),
    ('reach_depth_camera_v1 (depth cam, fixed base)', '1,999', 'v11 checkpoint', 'Learning'),
]
for row in training_rows:
    add_table_row(t2, row)
doc.add_paragraph()

doc.add_paragraph(
    'reach_v7 was trained from scratch for 2,600 iterations across 1,536 parallel environments. '
    'The robot learned to maintain balance first; falls (bad orientation termination) dropped '
    'from ~100% to <5% by iteration 400 without a fixed base. '
    'reach_v10 fine-tuned from v7 on a wider target randomization range and is the best policy, '
    'climbing to near-zero reward by iteration 4,000. '
    'reach_v11 fine-tuned further from v10 but plateaued at −4 despite continued training — '
    'likely a local minimum caused by the stricter target range. '
    'Action noise started near 1.0 for all runs and decayed within 500 iterations; '
    'reach_v10 uniquely re-explored after iteration 1,500, adapting to the wider target range.'
)

add_figure(doc, '01_reach_reward_comparison.png',
           'Figure 1: Mean episode reward comparison — reach_v7 (blue), reach_v10 (orange), reach_v11 (green).')
doc.add_paragraph()
add_figure(doc, '03_reach_v11_reward_breakdown.png',
           'Figure 2: Individual reward component breakdown for reach_v11.')
doc.add_paragraph()
add_figure(doc, '04_reach_v11_terminations.png',
           'Figure 3: Termination reason fractions for reach_v11 (bad orientation drops after iteration 700).')
doc.add_paragraph()

# ── 3.2 Stepping ────────────────────────────────────────────────────────────
add_heading(doc, '3.2 Stepping', 2)
doc.add_paragraph(
    'Goal: place each foot within 7 cm of a randomly sampled footstep target '
    'while maintaining upright stability and an alternating gait pattern.'
)

add_heading(doc, 'Observation Space', 3)
step_obs = [
    'base_lin_vel', 'base_ang_vel', 'base_rpy', 'joint_pos', 'joint_vel',
    'foot_contacts', 'actions', 'swing_foot', 'target_foot_pos_xy', 'base_xy_from_reset'
]
for o in step_obs:
    doc.add_paragraph(f'• {o}', style='List Bullet')

add_heading(doc, 'Reward Motivations', 3)
t3 = doc.add_table(rows=1, cols=2)
t3.style = 'Table Grid'
set_table_header(t3, ['Motivation', 'Reward / Shaping Terms'])
step_rewards = [
    ('Upright stability', 'torso_tilt_l2, root_height_below, termination_penalty, alive'),
    ('In-place behavior', 'base_lin_vel_xy_l2, base_ang_vel_z_l2, base_reset_position, base_reset_outward_vel, feet_midpoint_reset'),
    ('Alternating contact', 'alternating_contact_pattern, swing_foot_contact, feet_air_time'),
    ('Step-target tracking', 'step_target_reward, step_target_error, step_target_proximity'),
    ('Swing-leg lift', 'swing_knee_min_flex, swing_knee_angle_range, swing_leg_shortening, swing_foot_base_clearance, swing_support_height_difference, swing_support_height_excess, hip_only_swing'),
    ('Support-leg stability', 'support_knee_straight, support_foot_slip'),
    ('Foot ordering', 'feet_lateral_order'),
    ('Smooth whole-body', 'action_rate_l2, joint_vel_l2, joint_deviation_hip, joint_deviation_arms'),
]
for row in step_rewards:
    add_table_row(t3, row)
doc.add_paragraph()

add_heading(doc, 'Failure / Reset Conditions', 3)
doc.add_paragraph('• Timeout at 10.0 s')
doc.add_paragraph('• Orientation failure when root gravity-alignment angle exceeds 0.8 rad (45.8°)')
doc.add_paragraph('• Root height below 0.50 m')
doc.add_paragraph('• Contact force too high on base_link — illegal_contact(threshold=1.0)')

add_heading(doc, 'Training Progression', 3)
t4 = doc.add_table(rows=1, cols=3)
t4.style = 'Table Grid'
set_table_header(t4, ['Version', 'Main Change', 'Result'])
step_versions = [
    ('v1', 'Strong in-place + contact shaping', 'Feet barely moved. both_contact=0.994, swing=0.004'),
    ('v2', 'Weakened in-place penalty, stronger step/contact rewards', 'Feet lifted but too much. swing=0.855, clearance=0.353 m'),
    ('v3', 'Resumed from v2, added stronger target tracking', 'Foot placement improved. error=0.027 m, but bouncy. both_air=0.285'),
    ('v4', 'Resumed from v3, added upright/fall constraints', 'Transitional run'),
    ('v5', 'Trained from scratch with tuned upright objective', 'Final. error=0.0198 m, clearance=0.041 m, both_air≈0'),
]
for row in step_versions:
    add_table_row(t4, row)
doc.add_paragraph()
doc.add_paragraph('Final v5 active foot placement error: 0.0198 m (below the 7 cm threshold).')

# ── 3.3 Squatting ────────────────────────────────────────────────────────────
add_heading(doc, '3.3 Squatting', 2)
doc.add_paragraph(
    'Goal: track a commanded pelvis height within 15° torso tilt. '
    'The robot must transition between standing (~0.72 m) and deep squat (~0.35 m) '
    'on command while remaining stable.'
)

add_heading(doc, 'Observation Space', 3)
squat_obs = ['torso_tilt_l2', 'base_lin_vel_xy_l2', 'base_ang_vel_z_l2',
             'both_feet_contact', 'pelvis_height_error_l2', 'pelvis_height_tracking_exp', 'feet_slide']
for o in squat_obs:
    doc.add_paragraph(f'• {o}', style='List Bullet')

add_heading(doc, 'Failure Conditions', 3)
doc.add_paragraph('• Timeout | Orientation > 15° | Too low to ground | Contact forces too high')

add_heading(doc, 'Training Runs', 3)
t5 = doc.add_table(rows=1, cols=3)
t5.style = 'Table Grid'
set_table_header(t5, ['Run', 'Latest Checkpoint', 'Main Change'])
squat_runs = [
    ('16-25-46_final_squat_working', 'model_850.pt', 'Baseline squat: modest height tracking, no direct height-exp reward'),
    ('16-42-20_final_squat_working', 'model_9999.pt', 'Aggressive height matching added'),
    ('20-07-31_final_squat_resume', 'model_2499.pt', 'Expanded height range very aggressively'),
    ('20-45-28_final_squat_resume', 'model_2499.pt', 'Reduced max height from 0.85 to 0.75 m'),
    ('23-12-00_final_squat_resume', 'model_3498.pt', 'Stability-focused tuning'),
    ('23-42-17_final_squat_resume', 'model_4497.pt', 'Expanded stable range lower'),
    ('00-34-48_final_squat_resume', 'model_5000.pt', 'Continued same config as 23-42-17'),
]
for row in squat_runs:
    add_table_row(t5, row)
doc.add_paragraph()
doc.add_paragraph(
    'Training curves show pelvis height tracking improving rapidly after iteration 200, '
    'with mean episode length stabilizing at ~400 steps once the policy learns to balance '
    'while transitioning between heights.'
)

# ── 3.4 Walking ──────────────────────────────────────────────────────────────
add_heading(doc, '3.4 Walking', 2)
doc.add_paragraph(
    'Goal: track a target base velocity (up to 1.0 m/s forward) on flat and rough terrain, '
    'traveling ≥ 2 m in a straight line.'
)

add_heading(doc, 'Observation Space', 3)
walk_obs = [
    'base_lin_vel (3) + uniform noise [−0.1, 0.1]',
    'base_ang_vel (3) + uniform noise [−0.2, 0.2]',
    'projected_gravity (3) + uniform noise [−0.05, 0.05]',
    'velocity_commands (3) — generated 3-D base velocity target',
    'joint_pos (all joints, right-side sign-flipped) + noise [−0.01, 0.01]',
    'joint_vel (all joints, same convention) + noise [−1.5, 1.5]',
    'actions (previous, full joint dimension)',
    'height_scan (187) — downward ray-cast samples, clipped to [−1.0, 1.0] (rough terrain only)',
]
for o in walk_obs:
    doc.add_paragraph(f'• {o}', style='List Bullet')

doc.add_paragraph(
    'The locomotion policy was redesigned to limit locomotion responsibility to the lower body, '
    'with randomized upper-body motion and actuator gains. '
    'A harness-assisted curriculum was used: the robot was supported with a harness early in '
    'training, which was gradually removed as the policy became stable. '
    'Reward weights for forward speed (vx) and lateral/yaw penalties (vy/wz) were '
    'progressively strengthened, and small-command accuracy was tuned later in training. '
    'The robot successfully learned to walk on flat ground and demonstrated forward locomotion '
    'exceeding 2 m in evaluation.'
)

# ── Task F ──────────────────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, '4. Task F: High-Level Skills', 1)
doc.add_paragraph(
    'High-level skills combine locomotion and manipulation. '
    'We implemented two tasks: pushing a wall-mounted button and shooting a floating ball '
    'into a goal sphere. Both inherit from the flat locomotion base configuration.'
)

# ── 4.1 Push Button ──────────────────────────────────────────────────────────
add_heading(doc, '4.1 Push a Button', 2)
doc.add_paragraph(
    'The robot must: walk toward a wall-mounted push button, stop at a stand-off pose, '
    'reach with the right arm, and press the button plunger by ≥ 0.03 m.'
)

add_heading(doc, 'Why This Task is Hard', 3)
doc.add_paragraph(
    'The task combines two behaviors usually handled separately: whole-body locomotion '
    '(approaching the target) and arm manipulation (physical interaction). '
    'Full-body joint position control caused the walking policy to use the upper body, '
    'which conflicted with later manipulation tasks. '
    'The redesign limited locomotion responsibility to the lower body and '
    'randomized upper-body motion and actuator gains.'
)

add_heading(doc, 'Phase Decomposition', 3)
phases = [
    ('WALK', 'Keep the arm at its current pose while the base approaches the button'),
    ('APPROACH', 'Interpolate the end-effector to a point behind the button'),
    ('PRESS', 'Move the end effector through the plunger direction'),
    ('HOLD', 'Keep the end effector at the pressed target position'),
]
t6 = doc.add_table(rows=1, cols=2)
t6.style = 'Table Grid'
set_table_header(t6, ['Phase', 'Description'])
for row in phases:
    add_table_row(t6, row)
doc.add_paragraph()

add_heading(doc, 'Privileged Commands', 3)
doc.add_paragraph(
    'A privileged command vector [vx, vy, wz, target_height] computes a stand-off target '
    'in front of the button, converts the offset into the robot base frame, and clips commands:'
)
doc.add_paragraph('• Forward speed limit: 1.00 m/s')
doc.add_paragraph('• Backward speed limit: 0.15 m/s')
doc.add_paragraph('• Lateral speed limit: 0.25 m/s')
doc.add_paragraph('• Yaw speed limit: 1.00 rad/s')
doc.add_paragraph()
doc.add_paragraph('Success criterion: button_joint displacement ≥ 0.03 m.')

add_heading(doc, 'Training Strategy', 3)
doc.add_paragraph(
    'A harness curriculum removed support as training progressed. '
    'Reward weights for velocity tracking were strengthened axis-wise later in training, '
    'and a small-command penalty was introduced to improve fine-motor control near the button.'
)

# ── 4.2 Shoot a Ball ─────────────────────────────────────────────────────────
add_heading(doc, '4.2 Shoot a Ball into a Target', 2)
doc.add_paragraph(
    'The robot must use its right arm to push a gravity-disabled floating ball '
    '(radius 6 cm, mass 0.2 kg) into a goal sphere (radius 20 cm) placed 1.2 m in front. '
    'The ball starts at arm height near the right wrist (x ≈ 0.35 m, z ≈ 0.50 m). '
    'Only the five right-arm joints are actuated; the robot does not walk in this task.'
)

add_heading(doc, 'Scene Configuration', 3)
t7 = doc.add_table(rows=1, cols=2)
t7.style = 'Table Grid'
set_table_header(t7, ['Object', 'Properties'])
scene_rows = [
    ('Ball (orange sphere)', 'Radius=0.06 m, mass=0.2 kg, gravity disabled, linear_damping=0.02, spawn: x∈[0.28,0.42], y∈[−0.12,−0.02], z∈[0.42,0.58]'),
    ('Goal marker (green sphere)', 'Radius=0.20 m, kinematic=True, collision=False, placed at (1.20, 0.0, 0.50) m'),
]
for row in scene_rows:
    add_table_row(t7, row)
doc.add_paragraph()

add_heading(doc, 'Reward Structure', 3)
t8 = doc.add_table(rows=1, cols=3)
t8.style = 'Table Grid'
set_table_header(t8, ['Reward Term', 'Weight', 'Description'])
shoot_rewards = [
    ('arm_to_ball', '−2.0', 'L2 distance from right wrist to ball centre (reach the ball)'),
    ('ball_to_goal', '−1.5', 'XY distance from ball to goal — per-env using root_pos_w'),
    ('ball_velocity_toward_goal', '+3.0', 'Ball velocity component in the goal direction'),
    ('ball_at_goal', '+25.0', 'Binary bonus when ball enters goal (threshold 0.25 m)'),
    ('undesired_contacts', '−0.5', 'Penalizes non-foot body contacts'),
]
for row in shoot_rewards:
    add_table_row(t8, row)
doc.add_paragraph()

add_heading(doc, 'Observation Space', 3)
doc.add_paragraph(
    'Standard locomotion observations plus:'
)
doc.add_paragraph('• ball_pos_base (3) — ball position in robot base frame')
doc.add_paragraph('• goal_pos_base (3) — goal marker position in robot base frame')

add_heading(doc, 'Bug Diagnosis and Fixes', 3)
doc.add_paragraph(
    'Three critical bugs were identified by inspecting TensorBoard training curves:'
)
t9 = doc.add_table(rows=1, cols=3)
t9.style = 'Table Grid'
set_table_header(t9, ['Bug', 'Symptom', 'Fix'])
bugs = [
    ('v1: Leg joints active',
     'Robot collapsed every episode. Reward converged to −50 and never recovered.',
     'Restrict actions to RIGHT_ARM_JOINT_NAMES only (5 joints). Use reach_v11 checkpoint to pre-load balance.'),
    ('v3: Hardcoded world-frame GOAL_XY',
     'ball_to_goal used a fixed constant — only env_0 got the correct gradient. 1535/1536 envs received wrong reward. Policy flatlined near 0 (doing nothing got free reward).',
     'Read goal_marker.data.root_pos_w per-env at every step. Also add ball_velocity_toward_goal reward.'),
    ('v6: Ball spawns inside goal',
     'Ball spawn x∈[0.28,0.42] m, goal at x=0.70 m, threshold=0.35 m → overlap at episode start. Policy got free ball_at_goal bonus without doing anything.',
     'Move goal to 1.20 m. Reduce threshold to 0.25 m.'),
]
for row in bugs:
    add_table_row(t9, row)
doc.add_paragraph()

add_heading(doc, 'Training Results', 3)
doc.add_paragraph(
    'v4 (all bugs fixed) rapidly converged to reward ≈ +270 within 500 iterations '
    'and held stable for 2,000+ iterations. '
    'v7 (fine-tuned from reach_v11) stayed near 0 — the reach checkpoint provided '
    'good balance but the arm needed re-learning without the reaching reward. '
    'v4 is the final policy.'
)

add_figure(doc, '08_shoot_reward_comparison.png',
           'Figure 4: Shoot-ball training reward — all versions. v4 (green) converges to ~+270; v3 (goal bug) flatlines at 0; v1 (legs) crashes to −50.')
doc.add_paragraph()
add_figure(doc, '11_shoot_v4_reward_breakdown.png',
           'Figure 5: Individual reward components for shoot_ball_v4.')
doc.add_paragraph()
add_figure(doc, '12_shoot_bug_analysis.png',
           'Figure 6: Bug analysis — visual summary of the three identified bugs and their effect on training curves.')

# ── Task G ──────────────────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, '5. Task G: Isaac Lab Reinforcement Learning', 1)
doc.add_paragraph(
    'This section discusses the RL engineering choices that enabled successful training '
    'in Isaac Lab, covering observation design, reward shaping, domain randomization, '
    'and the depth-camera comparison.'
)

# ── 5.1 Observation Design ──────────────────────────────────────────────────
add_heading(doc, '5.1 Observation Design', 2)
doc.add_paragraph(
    'All policies share a common base observation (velocity, gravity, joint state, previous actions) '
    'with task-specific extensions. Task-relevant objects are expressed in the robot base frame '
    'via quat_apply_inverse, which handles arbitrary robot orientation without separate '
    'coordinate transforms.'
)
doc.add_paragraph()

t10 = doc.add_table(rows=1, cols=3)
t10.style = 'Table Grid'
set_table_header(t10, ['Task', 'Extra Observations', 'Rationale'])
obs_design = [
    ('Reaching (ground truth)', 'target_pos_base (3)', 'Direct 3-D target location; robot learns arm kinematics'),
    ('Reaching (depth camera)', 'generated_commands from DepthTargetPosCommand (3)', 'Smoothed (α=0.75) depth estimate; noisy but camera-realistic'),
    ('Shoot Ball', 'ball_pos_base (3), goal_pos_base (3)', 'Robot must know where to hit from (ball) and where to send it (goal)'),
    ('Left-Arm Reach', 'target_pos_base (3)', 'Same as right-arm reaching, but joint names changed to LEFT_ARM_JOINT_NAMES'),
]
for row in obs_design:
    add_table_row(t10, row)
doc.add_paragraph()

# ── 5.2 Reward Design ──────────────────────────────────────────────────────
add_heading(doc, '5.2 Reward Design', 2)
doc.add_paragraph(
    'Reward design follows a dense-shaping approach: '
    'a continuous gradient (distance or velocity) guides the policy toward the goal, '
    'while a large sparse bonus (ball_at_goal, reach_target) closes the final gap. '
    'Penalties on undesired contacts and joint deviations prevent degenerate behaviors. '
    'Key lessons learned:'
)
doc.add_paragraph('• Dense shaping alone (ball_to_goal distance) is insufficient if the policy can earn zero reward by doing nothing (requires per-env goal position).', style='List Bullet')
doc.add_paragraph('• A velocity-toward-goal reward (ball_velocity_toward_goal) is essential to teach the direction of the push, not just proximity.', style='List Bullet')
doc.add_paragraph('• The reach_target tanh reward creates a strong nonlinear gradient — reward spikes sharply within 8 cm of the target.', style='List Bullet')
doc.add_paragraph('• Large sparse bonuses (weight 25 for ball_at_goal) overwhelm noise and strongly shape terminal behavior.', style='List Bullet')

# ── 5.3 Domain Randomization ──────────────────────────────────────────────
add_heading(doc, '5.3 Domain Randomization', 2)
doc.add_paragraph(
    'Domain randomization is applied at episode reset via EventTerm configurations:'
)
t11 = doc.add_table(rows=1, cols=3)
t11.style = 'Table Grid'
set_table_header(t11, ['Task', 'Randomized Parameter', 'Range'])
rand_rows = [
    ('Reaching', 'Target sphere position (x, y, z)', 'x∈[0.30,0.50], y∈[−0.15,0.0], z∈[0.60,0.72] m'),
    ('Left-Arm Reach', 'Target position (y flipped)', 'x∈[0.30,0.50], y∈[0.05,0.20], z∈[0.60,0.72] m'),
    ('Shoot Ball', 'Ball spawn position (x, y, z)', 'x∈[0.28,0.42], y∈[−0.12,−0.02], z∈[0.42,0.58] m'),
    ('Walking', 'Robot base velocity command', 'Forward up to 1.0 m/s, lateral/yaw clipped'),
    ('Squatting', 'Target pelvis height', '0.35 m (squat) to 0.75 m (stand)'),
]
for row in rand_rows:
    add_table_row(t11, row)
doc.add_paragraph()
doc.add_paragraph(
    'Observation noise is also applied: joint positions ±0.01 rad, joint velocities ±1.5 rad/s, '
    'linear velocity ±0.1 m/s, angular velocity ±0.2 rad/s. '
    'Randomization of the target position for reach_v10 (wider range than v7) was '
    'the key factor that improved final reward from −2.8 to near 0.'
)

# ── 5.4 Depth Camera Comparison ─────────────────────────────────────────────
add_heading(doc, '5.4 Depth Camera Integration', 2)
doc.add_paragraph(
    'We trained a reaching policy using a robot-mounted depth camera '
    '(TiledCameraCfg, 64×64, pinhole, focal_length=20, range=[0.15, 3.0] m). '
    'The DepthTargetPosCommand reads the depth image each step, finds the nearest '
    'object in the workspace, and produces a smoothed 3-D position estimate '
    '(smooth_factor=0.75, resampled every 0.5 s) in the robot base frame. '
    'The policy receives this estimate — NOT the ground-truth position — via '
    'mdp.generated_commands("target_pos_base_from_depth"). '
    'The robot base is fixed (fix_base_link=True) so the policy only learns arm control.'
)

t12 = doc.add_table(rows=1, cols=3)
t12.style = 'Table Grid'
set_table_header(t12, ['Approach', 'Observation', 'Trade-off'])
depth_rows = [
    ('Ground-truth (reach_v10/v11)', 'Exact target_pos_base', 'Best performance; not realistic for deployment'),
    ('Depth camera (reach_depth_camera_v1)', 'Smoothed depth estimate via DepthTargetPosCommand', 'More realistic; noisier signal, smaller envs (512 vs 1536), requires --enable_cameras'),
]
for row in depth_rows:
    add_table_row(t12, row)
doc.add_paragraph()

doc.add_paragraph(
    'The perception pipeline uses depth-image segmentation to extract object clusters via BFS, '
    'scores each cluster as plane-like or sphere-like, and assigns a label. '
    'The 3-D centroid of the sphere-scored cluster is converted to the robot base frame '
    'and fed to the policy after temporal smoothing.'
)

add_figure(doc, '07_reach_depth_camera.png',
           'Figure 7: Depth-camera reaching policy training curve (reach_depth_camera_v1, fixed base, 1,999 iterations).')
doc.add_paragraph()

# ── 5.5 PPO Hyperparameters ─────────────────────────────────────────────────
add_heading(doc, '5.5 PPO Hyperparameters', 2)
t13 = doc.add_table(rows=1, cols=3)
t13.style = 'Table Grid'
set_table_header(t13, ['Parameter', 'Task Configs', 'Notes'])
ppo_rows = [
    ('Num environments', '1,536 (512 for depth cam)', 'GPU-parallel PhysX simulation'),
    ('Steps per env', '24', 'Rollout horizon'),
    ('Learning rate', '7.5e-4', 'Adaptive schedule'),
    ('Clip parameter ε', '0.2', 'Standard PPO clip'),
    ('Entropy coefficient', '0.005', 'Lower than default → less random exploration'),
    ('Gamma (γ)', '0.99', 'Long-horizon credit assignment'),
    ('Lambda (λ)', '0.95', 'GAE advantage estimation'),
    ('Network', '[256, 256, 128] actor/critic', 'ELU activations, no obs normalization'),
    ('Max iterations', '2000–4000', 'Task-dependent'),
]
for row in ppo_rows:
    add_table_row(t13, row)
doc.add_paragraph()

# ── Results Summary ──────────────────────────────────────────────────────────
add_heading(doc, '6. Results Summary', 1)
t14 = doc.add_table(rows=1, cols=4)
t14.style = 'Table Grid'
set_table_header(t14, ['Task', 'Target Metric', 'Achieved', 'Best Policy'])
results = [
    ('E: Reaching (right arm)', '≤ 5 cm to target', 'Near-zero reward (approaching threshold)', 'reach_v10, model_2350.pt'),
    ('E: Reaching (left arm)', '≤ 5 cm to target', 'Converged, fixed base', 'reach_left_fixed_v1, model_3999.pt'),
    ('E: Reaching (depth cam)', 'Camera-based target', 'Training, fixed base', 'reach_depth_camera_v1, model_1999.pt'),
    ('E: Stepping', '≤ 7 cm foot error', '0.0198 m foot error', 'Step v5'),
    ('E: Squatting', 'Torso tilt < 15°', 'Stable height tracking', 'Squat final resume'),
    ('E: Walking', '≥ 2 m forward', 'Robot walks on flat + rough ground', 'Walking flat/rough policy'),
    ('F: Push Button', 'button_joint ≥ 0.03 m', 'Push demonstrated', 'Push button v1'),
    ('F: Shoot Ball', 'Ball enters goal', 'Reward ≈ +270, stable', 'shoot_ball_v4, model_1999.pt'),
]
for row in results:
    add_table_row(t14, row)
doc.add_paragraph()

# ── Master Dashboard ──────────────────────────────────────────────────────────
add_figure(doc, '00_master_dashboard.png',
           'Figure 8: Master training dashboard — all tasks side by side (reach, left reach, depth camera, shoot ball).',
           width=6.0)
doc.add_paragraph()

# ── Conclusion ──────────────────────────────────────────────────────────────
add_heading(doc, '7. Conclusion', 1)
doc.add_paragraph(
    'We successfully trained reinforcement learning policies for six humanoid robot tasks '
    'in NVIDIA Isaac Lab. Key findings:'
)
doc.add_paragraph('• Iterative fine-tuning (v7 → v10) was more effective than long single-run training for the reaching task.', style='List Bullet')
doc.add_paragraph('• Per-env goal coordinates (reading root_pos_w at each step) are essential for multi-environment RL — hardcoded world coordinates break gradients for all but one environment.', style='List Bullet')
doc.add_paragraph('• A velocity-toward-goal shaping reward is necessary to teach directional pushing, not just proximity.', style='List Bullet')
doc.add_paragraph('• The depth camera pipeline produces a noisier but physically realistic target estimate; the policy learns to cope with the noise via temporal smoothing (α=0.75).', style='List Bullet')
doc.add_paragraph('• Curriculum learning (harness removal, progressive reward weighting) enabled stable walking and button pushing, which combine locomotion and manipulation.', style='List Bullet')
doc.add_paragraph('• The stepping task achieved 0.0198 m foot placement error (below the 7 cm target) using a rich set of shaping rewards covering all aspects of gait.', style='List Bullet')

# ── Save ────────────────────────────────────────────────────────────────────
out_path = '/home/wanglab22/CSCE50103-IsaacLab/final_report.docx'
doc.save(out_path)
print(f'Report saved to: {out_path}')
