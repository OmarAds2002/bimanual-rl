# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Config for the bimanual UR10 cylinder-handoff task (DirectRLEnv).

Layout (side view, x-z plane):

        handoff point (0, 0, 0.85)
              v
   [arm1]           [arm2]      <- both bases bolted to the table TOP (z = 0.5)
  ===========================   <- table top at z = 0.5
  |          TABLE         |    <- static cuboid, z from 0.0 -> 0.5
  ===========================
  ---------------------------   <- ground plane

The cylinder starts upright on the table in front of arm1, placed deep enough on
arm1's side that arm2 physically can't reach it. That's deliberate: it forces a
real handoff instead of letting arm2 just grab the object off the table.
"""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR10_CFG

# UR10 revolute joints in kinematic order (base -> tool).
UR10_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Elbow-up "ready" pose both arms start in (radians).
_READY_POSE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.712,
    "elbow_joint": 1.712,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
}

# --------------------------------------------------------------------------- #
#  Layout geometry                                                            #
# --------------------------------------------------------------------------- #
_BASE_X = 0.7                   # half the base-to-base separation
_TABLE_TOP = 0.5               # table surface height; arms mount on top of this
_CYL_HALF_H = 0.10             # cylinder half-height (spawn height = 0.20)

# Where the object gets handed over: above the table, centred between the arms.
_HANDOFF_POS = (0.0, 0.0, 0.85)

# Arm2 is spun 180 deg about +Z so it faces arm1 (opposed layout).
_ROT_FACING = (0.0, 0.0, 0.0, 1.0)   # (w, x, y, z)
_ROT_DEFAULT = (1.0, 0.0, 0.0, 0.0)  # identity

# The stock UR10_CFG ships with self-collisions OFF, which let arm2 fold its wrist
# straight through its own upper arm for free (PhysX just doesn't compute those
# contacts). Turn it on. deepcopy first so we don't mutate the shared global cfg,
# and the stock spawn has no articulation_props at all, so we create one.
_UR10_SELFCOL = copy.deepcopy(UR10_CFG)
_UR10_SELFCOL.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
)


@configclass
class HandoffEnvCfg(DirectRLEnvCfg):
    """Bimanual UR10 cylinder-handoff environment config."""

    # --- env timing (required) ---
    decimation: int = 4
    episode_length_s: float = 6.0

    # Camera used by the viewer AND by --video recording. eye = camera position,
    # lookat = the point it aims at, both in world metres. Aimed at the handoff point
    # (0, 0, 0.85) between the two arms. Tweak eye to reframe the shot.
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.5, 2.5, 1.8),
        lookat=(0.0, 0.0, 0.85),
    )

    # --- settle / posture thresholds (read by the reward) ---
    settle_speed_thresh: float = 0.5   # rad/s per-arm joint-vel norm to count as "quiet"
    arm1_home_thresh: float = 0.8      # rad, L2 joint deviation to count as "home"
    w_joint_limit: float = 1.0
    w_posture: float = 0.01            # small on purpose: arm1 HAS to leave home to reach
    w_action_rate: float = 0.02
    w_contact: float = 0.005           # only matters if contact sensors get wired up

    # --- spaces (required) ---
    # 12 arm joints + 2 grasp-intent scalars = 14
    action_space: int = 14
    # 24 joints (12 pos + 12 vel over both arms) + 13 object state (7 pose + 6 vel)
    #  + 6 obj->EE vectors + 6 arm->grasp-target vectors (give + recv) + 2 grasp flags = 51
    observation_space: int = 51
    state_space: int = 0

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # ----------------------------------------------------------------------- #
    #  Entities                                                               #
    # ----------------------------------------------------------------------- #
    # Arm1 (giver): mounted on the table top at -x, facing +x toward the centre.
    robot_1: ArticulationCfg = _UR10_SELFCOL.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-_BASE_X, 0.0, _TABLE_TOP),
            rot=_ROT_DEFAULT,
            joint_pos=_READY_POSE,
        ),
    )

    # Arm2 (receiver): mounted at +x, rotated to face arm1.
    robot_2: ArticulationCfg = _UR10_SELFCOL.replace(
        prim_path="/World/envs/env_.*/Robot_2",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(_BASE_X, 0.0, _TABLE_TOP),
            rot=_ROT_FACING,
            joint_pos=_READY_POSE,
        ),
    )

    # The cylinder: starts upright on the table in front of arm1.
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.20,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.35, 0.1),
                metallic=0.1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-_BASE_X, 0.3, _TABLE_TOP + _CYL_HALF_H + 0.005)
        ),
    )

    # Static table: fixed collision surface the arms sit on. rigid_props=None makes
    # it a static collider.
    table_cfg: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
        size=(2.8, 1.4, _TABLE_TOP),
        rigid_props=None,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
    )
    # Table CENTER position when spawned (so the top lands at _TABLE_TOP).
    table_pos: tuple[float, float, float] = (0.0, 0.0, _TABLE_TOP / 2.0)

    # --- ground + light ---
    terrain_cfg: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg()
    dome_light_cfg: sim_utils.DomeLightCfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.9, 0.9, 0.9),
    )

    # ----------------------------------------------------------------------- #
    #  Task knobs read by the env logic                                       #
    # ----------------------------------------------------------------------- #
    ee_body_name: str = "ee_link"
    arm_joint_names: list[str] = UR10_ARM_JOINTS

    handoff_pos: tuple[float, float, float] = _HANDOFF_POS
    grasp_dist_threshold: float = 0.10
    drop_height: float = 0.45
    reset_object_xy_range: float = 0.03
    action_scale: float = 0.5

    # Elongated "baton" so two wrists can grab opposite ends without overlapping.
    cyl_length: float = 0.38                                      # long-axis length (m)
    cyl_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)  # long axis in body frame

    # Schmitt-trigger grasp latch: intent has to exceed grasp_on to LATCH and stay
    # latched until it drops below grasp_off. Stops noisy actions from flicking the
    # grasp on/off every couple of steps.
    grasp_on: float = 0.5
    grasp_off: float = -0.5

    # --- optional start-in-hand curriculum (OFF by default) ---
    # Reset a fraction of envs already holding + lifting the object, anneal to 0.
    # Turned out to be unnecessary once the geometry was fixed, so it's disabled.
    # Flip use_start_in_hand to True to try it.
    use_start_in_hand: bool = False
    curriculum_start_prob: float = 1.0    # p(in-hand) at step 0, only used if the toggle is on
    curriculum_end_step: int = 30_000_000  # env-steps over which p anneals 1.0 -> 0.0