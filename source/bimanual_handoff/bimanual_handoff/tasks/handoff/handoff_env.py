# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Bimanual UR10 cylinder-handoff environment (DirectRLEnv).

Two opposed UR10 arms hand a cylinder across a shared table. Arm1 grasps from its
side, carries it to a central handoff point, and transfers it to arm2. There are no
fingers — a grasp is a kinematic weld (plain UR10, no gripper in this Isaac Lab
build). The object starts deep on arm1's side so arm2 can't cheat and grab it
directly; the handoff is forced.

The design here is the result of a long debug arc. The short version of what took
me a while to figure out:

  * The weld used to snap the object's center onto the holder's EE origin. That made
    "arm2 reaches the object" mean "two UR10 wrists occupy the same 10 cm" — which is
    interpenetration, not a grasp. Now the weld captures the object's pose *relative*
    to the EE at latch time and maintains that transform, so the object is held where
    it was actually touched.
  * The cylinder is long (0.38 m) with give/recv sites at each end. Arm1 grabs the
    body, arm2 aims for the far end, so the two wrists never fight for the same spot.
  * Grasps latch with a Schmitt trigger instead of following raw noisy action values.
  * The stage ladder is monotone with a mutual-grasp rung (both hold) between
    "arrived" and "done", so completing the handoff is never worse than just parking.
  * arm2 gets a bounded tanh dense reward that never dries out (unlike a best-so-far
    well), so it always has a gradient to follow.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)

from .handoff_env_cfg import HandoffEnvCfg


class HandoffEnv(DirectRLEnv):
    cfg: HandoffEnvCfg

    def __init__(self, cfg: HandoffEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve constant indices once (articulations exist after super()/_setup_scene).
        self._arm1_joint_ids, _ = self.robot_1.find_joints(self.cfg.arm_joint_names)
        self._arm2_joint_ids, _ = self.robot_2.find_joints(self.cfg.arm_joint_names)
        self._ee1_body_id, _ = self.robot_1.find_bodies(self.cfg.ee_body_name)
        self._ee2_body_id, _ = self.robot_2.find_bodies(self.cfg.ee_body_name)

        # Which arm currently holds the object (per env).
        self._held_by_1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._held_by_2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Previous-step held flags, so we can detect the exact frame of a release.
        self._prev_held_by_1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_held_by_2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Grasp offsets: the object's pose in the holder's EE frame, captured the
        # frame a grasp latches. The weld rebuilds the object pose from these each
        # substep, so the object stays where it was grabbed instead of teleporting
        # onto the wrist origin. Identity = object coincident with the EE.
        self._off_p1 = torch.zeros(self.num_envs, 3, device=self.device)
        self._off_p2 = torch.zeros(self.num_envs, 3, device=self.device)
        self._off_q1 = torch.zeros(self.num_envs, 4, device=self.device)
        self._off_q2 = torch.zeros(self.num_envs, 4, device=self.device)
        self._off_q1[:, 0] = 1.0  # (w,x,y,z) identity
        self._off_q2[:, 0] = 1.0

        # Handoff position as a tensor (env-local coords).
        self._handoff_pos = torch.tensor(
            self.cfg.handoff_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, 3)

        # Cylinder long axis in body frame, used to place the give/recv end sites.
        self._cyl_axis = torch.tensor(
            self.cfg.cyl_axis_local, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, 3)
        self._site_half = 0.3 * self.cfg.cyl_length  # sites at +/-30% along the axis

        # Table-top height, for the "lifted" reward gate.
        self._table_top_z = self.cfg.table_pos[2] + self.cfg.table_cfg.size[2] / 2.0

        # Action-rate history (init lazily on the first step to match the action dim).
        self._prev_actions: torch.Tensor | None = None

        # Best-distance buffers for arm1's potential-based shaping. arm2 doesn't need
        # one anymore — it uses a bounded dense reward that never dries up.
        self._best_d_ee1_give = torch.full((self.num_envs,), 999.0, device=self.device)
        self._best_d_obj_handoff = torch.full((self.num_envs,), 999.0, device=self.device)

    # ------------------------------------------------------------------ #
    #  Scene                                                             #
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        # (a) turn cfgs into live objects
        self.robot_1 = Articulation(self.cfg.robot_1)
        self.robot_2 = Articulation(self.cfg.robot_2)
        self.object = RigidObject(self.cfg.object)

        # (b) ground plane
        self.cfg.terrain_cfg.func("/World/ground", self.cfg.terrain_cfg)

        # (c) clone the template into num_envs copies
        self.scene.clone_environments(copy_from_source=False)

        # (d) static table under/around the arms
        self.cfg.table_cfg.func(
            "/World/envs/env_.*/Table",
            self.cfg.table_cfg,
            translation=self.cfg.table_pos,
        )

        # (e) register with the scene so the framework tracks/steps them
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2
        self.scene.rigid_objects["object"] = self.object

        # light
        self.cfg.dome_light_cfg.func("/World/Light", self.cfg.dome_light_cfg)

    # ------------------------------------------------------------------ #
    #  Actions                                                           #
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        if self._prev_actions is None:
            self._prev_actions = torch.zeros_like(self.actions)

        # split: [0:6] arm1 joints, [6:12] arm2 joints, [12:14] grasp intents
        a1 = actions[:, 0:6]
        a2 = actions[:, 6:12]
        self._grasp_intent = actions[:, 12:14]

        # scaled delta from the default pose (action 0 => hold ready pose)
        self._targets_1 = (
            self.robot_1.data.default_joint_pos[:, self._arm1_joint_ids]
            + self.cfg.action_scale * a1
        )
        self._targets_2 = (
            self.robot_2.data.default_joint_pos[:, self._arm2_joint_ids]
            + self.cfg.action_scale * a2
        )

        # Clamp targets to the soft joint limits. Raw actions are unbounded Gaussians;
        # without this, default + 0.5*a could push a target way past the joint limit
        # and the arm would just slam into the stop and sit there.
        lim1 = self.robot_1.data.soft_joint_pos_limits[:, self._arm1_joint_ids, :]
        self._targets_1 = torch.clamp(self._targets_1, lim1[..., 0], lim1[..., 1])
        lim2 = self.robot_2.data.soft_joint_pos_limits[:, self._arm2_joint_ids, :]
        self._targets_2 = torch.clamp(self._targets_2, lim2[..., 0], lim2[..., 1])

        # evaluate grasp intents and update the held flags
        self._apply_grasp()

    def _apply_action(self) -> None:
        self.robot_1.set_joint_position_target(self._targets_1, joint_ids=self._arm1_joint_ids)
        self.robot_2.set_joint_position_target(self._targets_2, joint_ids=self._arm2_joint_ids)

        # kinematic weld: rebuild the object pose from the captured EE-relative offset
        self._weld_held_object()

    # ------------------------------------------------------------------ #
    #  Give / receive end sites                                          #
    # ------------------------------------------------------------------ #
    def _end_sites(self, obj_pos_w: torch.Tensor, obj_quat_w: torch.Tensor):
        """World positions of the two ends of the cylinder.

        give_site is arm1's end, recv_site is arm2's. Splitting the long axis is what
        keeps the two wrists apart: arm1 grabs the body, arm2 aims for the far end,
        so they never need to be in the same 10 cm ball.
        """
        axis_w = quat_apply(obj_quat_w, self._cyl_axis.expand(obj_quat_w.shape[0], 3))
        give_site = obj_pos_w - axis_w * self._site_half
        recv_site = obj_pos_w + axis_w * self._site_half
        return give_site, recv_site

    # ------------------------------------------------------------------ #
    #  Grasp / weld                                                      #
    # ------------------------------------------------------------------ #
    def _capture_offset(self, mask, ee_pos, ee_quat, buf_p, buf_q) -> None:
        """Record the object's pose in the EE frame for the envs in `mask`.

        Called the frame a new grasp latches, so the weld can hold the object exactly
        where it was at contact instead of snapping it to the EE origin.
        """
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        obj_p = self.object.data.root_pos_w[ids]
        obj_q = self.object.data.root_quat_w[ids]
        buf_p[ids] = quat_rotate_inverse(ee_quat[ids], obj_p - ee_pos[ids])
        buf_q[ids] = quat_mul(quat_conjugate(ee_quat[ids]), obj_q)

    def _apply_grasp(self) -> None:
        """Update the latched grasp flags. Runs once per RL step.

        A grasp latches ON when intent > grasp_on AND the EE is in range, and stays
        on until intent < grasp_off (a deliberate, sustained release). The Schmitt
        deadband kills the old failure where noisy intent flipped the grasp every
        couple of steps and made every transfer a near-certain drop.

        arm1 is NOT stripped when arm2 grabs — both holding is a valid state (the
        mutual-grasp rung). arm1 only lets go when it commands a release, which the
        ladder rewards. That's what removes the reward cliff.
        """
        ee1_pos = self.robot_1.data.body_pos_w[:, self._ee1_body_id[0]]    # (N, 3)
        ee2_pos = self.robot_2.data.body_pos_w[:, self._ee2_body_id[0]]    # (N, 3)
        ee1_quat = self.robot_1.data.body_quat_w[:, self._ee1_body_id[0]]  # (N, 4)
        ee2_quat = self.robot_2.data.body_quat_w[:, self._ee2_body_id[0]]  # (N, 4)
        obj_pos = self.object.data.root_pos_w                              # (N, 3)
        obj_quat = self.object.data.root_quat_w                            # (N, 4)

        _, recv_site = self._end_sites(obj_pos, obj_quat)

        dist1 = torch.norm(ee1_pos - obj_pos, dim=-1)          # arm1 grabs anywhere on the body
        dist2_recv = torch.norm(ee2_pos - recv_site, dim=-1)   # arm2 targets the free end

        intent1 = self._grasp_intent[:, 0]
        intent2 = self._grasp_intent[:, 1]

        handoff_w = self._handoff_pos + self.scene.env_origins  # (N, 3)
        near_handoff = torch.norm(obj_pos[:, :2] - handoff_w[:, :2], dim=-1) < 0.20

        close1 = dist1 < self.cfg.grasp_dist_threshold
        close2 = dist2_recv < self.cfg.grasp_dist_threshold

        acquire1 = (intent1 > self.cfg.grasp_on) & close1
        release1 = intent1 < self.cfg.grasp_off
        acquire2 = (intent2 > self.cfg.grasp_on) & close2 & near_handoff
        release2 = intent2 < self.cfg.grasp_off

        # snapshot before mutating (used for the release velocity below)
        self._prev_held_by_1 = self._held_by_1.clone()
        self._prev_held_by_2 = self._held_by_2.clone()

        free = ~self._held_by_1 & ~self._held_by_2
        new_grab_1 = acquire1 & free                                 # fresh grab off the table
        new_grab_2 = acquire2 & self._held_by_1 & ~self._held_by_2   # receiver takes over (mutual)

        # latch update
        self._held_by_1 = (self._held_by_1 | new_grab_1) & ~release1
        self._held_by_2 = (self._held_by_2 | new_grab_2) & ~release2

        # capture EE-relative offsets for grasps that actually latched this frame
        self._capture_offset(new_grab_1 & self._held_by_1, ee1_pos, ee1_quat, self._off_p1, self._off_q1)
        self._capture_offset(new_grab_2 & self._held_by_2, ee2_pos, ee2_quat, self._off_p2, self._off_q2)

        # On release, stamp the EE velocity onto the object so it doesn't just freeze.
        just_released_1 = self._prev_held_by_1 & ~self._held_by_1
        rel1_ids = just_released_1.nonzero(as_tuple=False).squeeze(-1)
        if rel1_ids.numel() > 0:
            ee1_vel = self.robot_1.data.body_vel_w[:, self._ee1_body_id[0]]  # (N, 6)
            self.object.write_root_velocity_to_sim(ee1_vel[rel1_ids], env_ids=rel1_ids)

        just_released_2 = self._prev_held_by_2 & ~self._held_by_2
        rel2_ids = just_released_2.nonzero(as_tuple=False).squeeze(-1)
        if rel2_ids.numel() > 0:
            ee2_vel = self.robot_2.data.body_vel_w[:, self._ee2_body_id[0]]  # (N, 6)
            self.object.write_root_velocity_to_sim(ee2_vel[rel2_ids], env_ids=rel2_ids)

    def _weld_held_object(self) -> None:
        """Rebuild the object pose from the holder's EE frame every substep.

        target = ee_pose ∘ captured_offset (relative transform, NOT a center-snap).
        When both arms hold, arm1 wins the overlap (it's still the giver). The instant
        arm1 releases, control passes to arm2 through arm2's own captured offset, so
        there's no jump in the object pose across the transfer.
        """
        held_any = self._held_by_1 | self._held_by_2
        if not held_any.any():
            return

        ee1_pos = self.robot_1.data.body_pos_w[:, self._ee1_body_id[0]]
        ee2_pos = self.robot_2.data.body_pos_w[:, self._ee2_body_id[0]]
        ee1_quat = self.robot_1.data.body_quat_w[:, self._ee1_body_id[0]]
        ee2_quat = self.robot_2.data.body_quat_w[:, self._ee2_body_id[0]]

        target_pos = self.object.data.root_pos_w.clone()    # (N, 3)
        target_quat = self.object.data.root_quat_w.clone()  # (N, 4)

        # arm2 is the sole holder -> follow arm2
        h2_only = self._held_by_2 & ~self._held_by_1
        if h2_only.any():
            target_pos[h2_only] = ee2_pos[h2_only] + quat_apply(ee2_quat[h2_only], self._off_p2[h2_only])
            target_quat[h2_only] = quat_mul(ee2_quat[h2_only], self._off_q2[h2_only])

        # arm1 holds (including mutual grasp) -> follow arm1, wins the overlap
        h1 = self._held_by_1
        if h1.any():
            target_pos[h1] = ee1_pos[h1] + quat_apply(ee1_quat[h1], self._off_p1[h1])
            target_quat[h1] = quat_mul(ee1_quat[h1], self._off_q1[h1])

        zero_vel = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        root_state = torch.cat([target_pos, target_quat, zero_vel], dim=-1)  # (N, 13)

        held_ids = held_any.nonzero(as_tuple=False).squeeze(-1)
        self.object.write_root_state_to_sim(root_state[held_ids], env_ids=held_ids)

    # ------------------------------------------------------------------ #
    #  Observations (51 dims — keep cfg.observation_space in sync)        #
    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict:
        # per-arm joint state (12 each = 24)
        j1_pos = self.robot_1.data.joint_pos[:, self._arm1_joint_ids]
        j1_vel = self.robot_1.data.joint_vel[:, self._arm1_joint_ids]
        j2_pos = self.robot_2.data.joint_pos[:, self._arm2_joint_ids]
        j2_vel = self.robot_2.data.joint_vel[:, self._arm2_joint_ids]

        # object state (7 pose + 6 vel = 13); position is env-local
        obj_pos_w = self.object.data.root_pos_w
        obj_quat = self.object.data.root_quat_w
        obj_pos = obj_pos_w - self.scene.env_origins
        obj_lvel = self.object.data.root_lin_vel_w
        obj_avel = self.object.data.root_ang_vel_w

        # EE -> object-center vectors (6)
        ee1_pos = self.robot_1.data.body_pos_w[:, self._ee1_body_id[0]]
        ee2_pos = self.robot_2.data.body_pos_w[:, self._ee2_body_id[0]]
        obj_to_ee1 = ee1_pos - obj_pos_w
        obj_to_ee2 = ee2_pos - obj_pos_w

        # arm2 -> its actual grasp target (recv end) and arm1 -> its give-end target.
        # Without these each arm couldn't see where it was supposed to grab and its
        # policy was basically flying blind.
        give_site, recv_site = self._end_sites(obj_pos_w, obj_quat)
        ee1_to_give = give_site - ee1_pos
        ee2_to_recv = recv_site - ee2_pos

        # grasp flags (2)
        grasp = torch.stack([self._held_by_1, self._held_by_2], dim=1).float()

        obs = torch.cat(
            [j1_pos, j1_vel, j2_pos, j2_vel,           # 24
             obj_pos, obj_quat, obj_lvel, obj_avel,    # 13
             obj_to_ee1, obj_to_ee2,                   # 6
             ee1_to_give, ee2_to_recv,                 # 6
             grasp],                                   # 2
            dim=-1,
        )  # total = 51
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    #  Reward                                                            #
    # ------------------------------------------------------------------ #
    def _get_rewards(self) -> torch.Tensor:
        # ---------------- shared quantities ---------------- #
        ee1_pos = self.robot_1.data.body_pos_w[:, self._ee1_body_id[0]]   # (N, 3)
        ee2_pos = self.robot_2.data.body_pos_w[:, self._ee2_body_id[0]]   # (N, 3)
        obj_pos_w = self.object.data.root_pos_w                           # (N, 3)
        obj_quat_w = self.object.data.root_quat_w                         # (N, 4)
        obj_pos_local = obj_pos_w - self.scene.env_origins               # (N, 3)
        obj_z = obj_pos_local[:, 2]                                       # (N,)

        give_site, recv_site = self._end_sites(obj_pos_w, obj_quat_w)
        dist_ee1_give = torch.norm(ee1_pos - give_site, dim=-1)          # (N,)
        dist_ee2_recv = torch.norm(ee2_pos - recv_site, dim=-1)          # (N,)
        dist_ee1_obj = torch.norm(ee1_pos - obj_pos_w, dim=-1)           # logging only

        handoff_w = self._handoff_pos + self.scene.env_origins           # (N, 3)
        dist_obj_handoff_xy = torch.norm(obj_pos_w[:, :2] - handoff_w[:, :2], dim=-1)

        # joint state for both arms (settle + posture + limit terms)
        j1_pos = self.robot_1.data.joint_pos[:, self._arm1_joint_ids]    # (N, 6)
        j2_pos = self.robot_2.data.joint_pos[:, self._arm2_joint_ids]    # (N, 6)
        j1_vel = self.robot_1.data.joint_vel[:, self._arm1_joint_ids]
        j2_vel = self.robot_2.data.joint_vel[:, self._arm2_joint_ids]
        default1 = self.robot_1.data.default_joint_pos[:, self._arm1_joint_ids]
        default2 = self.robot_2.data.default_joint_pos[:, self._arm2_joint_ids]

        speed1 = torch.norm(j1_vel, dim=-1)                              # (N,)
        speed2 = torch.norm(j2_vel, dim=-1)                              # (N,)
        dev1 = torch.norm(j1_pos - default1, dim=-1)                     # (N,) arm1 from home
        dev2 = torch.norm(j2_pos - default2, dim=-1)                     # (N,) arm2 from home

        # ================================================================ #
        #  MONOTONE STAGE LADDER (must be mutually exclusive)               #
        #                                                                  #
        #  s2 grasped-on-table 0.5                                         #
        #  s3 carrying         0.75                                        #
        #  s4 arrived          1.0                                         #
        #  s5 mutual grasp     2.0   both hold, no drop risk mid-transfer   #
        #  s6 handover done    4.0   strictly better annuity than parking   #
        #  s7 settled          6.0   done + both arms quiet + arm1 home     #
        #                                                                  #
        #  The bug I hit: s7 is a SUBSET of s6, so assigning reward[s6]     #
        #  after reward[s7] silently overwrote the settle bonus back to     #
        #  4.0 and the policy had zero reason to ever stop moving. Fix is    #
        #  to make the masks disjoint BEFORE assigning anything.            #
        # ================================================================ #
        held_1 = self._held_by_1
        held_2 = self._held_by_2
        lifted = obj_z > (self._table_top_z + 0.05)
        obj_near_handoff = dist_obj_handoff_xy < 0.20
        dropped = obj_z < self.cfg.drop_height

        both_quiet = (speed1 < self.cfg.settle_speed_thresh) & (speed2 < self.cfg.settle_speed_thresh)
        arm1_home = dev1 < self.cfg.arm1_home_thresh

        s1 = ~held_1 & ~held_2 & ~dropped                   # reaching
        s2 = held_1 & ~held_2 & ~lifted                     # grasped, still on table
        s3 = held_1 & ~held_2 & lifted & ~obj_near_handoff  # carrying
        s4 = held_1 & ~held_2 & lifted & obj_near_handoff   # arrived
        s5 = held_1 & held_2 & lifted                       # mutual grasp
        s6_raw = held_2 & ~held_1 & lifted                  # handover complete (any)
        s7 = s6_raw & both_quiet & arm1_home                # settled
        s6 = s6_raw & ~s7                                   # <-- disjoint: s6 excludes s7

        reward = torch.zeros(self.num_envs, device=self.device)
        reward[s2] = 0.5
        reward[s3] = 0.75
        reward[s4] = 1.0
        reward[s5] = 2.0
        reward[s6] = 4.0
        reward[s7] = 6.0
        reward[dropped] = -2.0   # softened from -5.0; drops are rare with latching and a
        #                          smaller cliff is easier for the critic to bootstrap

        # ---------------- potential-based progress (arm1 only) ---------------- #
        # Only pays for beating this episode's best, so it can't be farmed by sitting
        # still. Kept for arm1 because it worked; arm2 uses the dense tanh below.
        def progress(dist, best_buf, active_mask, scale):
            improved = (best_buf - dist).clamp(min=0.0, max=0.15)  # 0.15 clamp kills the
            #                                                        999-sentinel phantom spike
            r = scale * improved * active_mask.float()
            new_best = torch.minimum(best_buf, dist)
            best_buf[active_mask] = new_best[active_mask]
            return r

        r_reach1 = progress(dist_ee1_give, self._best_d_ee1_give, s1, scale=3.0)
        phase_1 = held_1 & ~held_2
        r_transport = progress(dist_obj_handoff_xy, self._best_d_obj_handoff, phase_1, scale=3.0)

        # ---------------- arm2 bounded dense reward (robosuite form) ---------------- #
        # Gated so arm2 isn't dragged into arm1 during the carry; capped at 0.5 so it
        # can't be farmed; tanh so it never dries and needs no target state machine.
        arm2_gate = (phase_1 & obj_near_handoff) | s5
        r_arm2 = 0.5 * (1.0 - torch.tanh(5.0 * dist_ee2_recv)) * arm2_gate.float()

        # ---------------- settle shaping (dense path into s7) ---------------- #
        # All gated on the handover being complete, so they never fight the fast motion
        # arm2 needs during the approach.
        settled_gate = s6_raw.float()
        # exp(-0.35*speed), NOT exp(-2*speed): the tighter version underflowed to ~1e-8
        # at 4-5 rad/s, so there was literally no gradient to descend. 0.35 keeps a
        # usable slope while the arms are still moving.
        r_settle = 1.5 * torch.exp(-0.35 * (speed1 + speed2)) * settled_gate
        r_quiet = -0.15 * (speed1 + speed2) * settled_gate   # constant slope everywhere
        r_arm1_home = 1.0 * (1.0 - torch.tanh(2.0 * dev1)) * settled_gate

        # ---------------- posture / collision avoidance ---------------- #
        # (a) Soft joint-limit overshoot penalty. Isaac's soft limits sit just inside
        #     the hard stops; penalising overshoot keeps the policy off the stops, which
        #     is where the folded self-colliding configs live.
        lim1 = self.robot_1.data.soft_joint_pos_limits[:, self._arm1_joint_ids, :]  # (N,6,2)
        lim2 = self.robot_2.data.soft_joint_pos_limits[:, self._arm2_joint_ids, :]
        over1 = (lim1[..., 0] - j1_pos).clamp(min=0.0) + (j1_pos - lim1[..., 1]).clamp(min=0.0)
        over2 = (lim2[..., 0] - j2_pos).clamp(min=0.0) + (j2_pos - lim2[..., 1]).clamp(min=0.0)
        r_limits = -self.cfg.w_joint_limit * (over1.sum(dim=-1) + over2.sum(dim=-1))

        # (b) Weak pull toward the ready pose. Small on purpose — arm1 MUST leave home
        #     to reach — so this only breaks ties between equivalent configs, nudging
        #     toward the natural elbow-up one instead of a folded wrist.
        r_posture = -self.cfg.w_posture * (
            torch.sum((j1_pos - default1) ** 2, dim=-1)
            + torch.sum((j2_pos - default2) ** 2, dim=-1)
        )

        # (c) Real self-collision penalty IF contact sensors get wired. Without
        #     enabled_self_collisions=True this can never fire — PhysX doesn't compute
        #     intra-arm contacts, which is exactly why the folded pose used to be free.
        if getattr(self, "_contact_sensor_1", None) is not None:
            f1 = torch.norm(self._contact_sensor_1.data.net_forces_w, dim=-1).sum(dim=-1)
            f2 = torch.norm(self._contact_sensor_2.data.net_forces_w, dim=-1).sum(dim=-1)
            r_contact = -self.cfg.w_contact * (f1 + f2).clamp(max=50.0)
        else:
            r_contact = torch.zeros(self.num_envs, device=self.device)

        # ---------------- action-rate penalty ---------------- #
        if self._prev_actions is None:
            r_rate = torch.zeros(self.num_envs, device=self.device)
        else:
            r_rate = -self.cfg.w_action_rate * torch.sum(
                (self.actions - self._prev_actions) ** 2, dim=-1
            )
        self._prev_actions = self.actions.clone()

        # ---------------- total ---------------- #
        reward = (
            reward
            + r_reach1
            + r_transport
            + r_arm2
            + r_settle
            + r_arm1_home
            + r_limits
            + r_posture
            + r_contact
            + r_rate
            + r_quiet
        )

        # ---------------- per-component logging ---------------- #
        self.extras["log"] = {
            "reward_mean": reward.mean().item(),
            "reward_max": reward.max().item(),
            "stage_1_frac": s1.float().mean().item(),
            "stage_2_frac": s2.float().mean().item(),
            "stage_3_frac": s3.float().mean().item(),
            "stage_4_frac": s4.float().mean().item(),
            "stage_5_frac": s5.float().mean().item(),      # mutual grasp
            "stage_6_frac": s6.float().mean().item(),      # handover, still moving
            "stage_7_frac": s7.float().mean().item(),      # settled
            "success_frac": s6_raw.float().mean().item(),  # s6 + s7 (any handover)
            "drop_frac": dropped.float().mean().item(),
            "prog_reach1": r_reach1.mean().item(),
            "prog_transport": r_transport.mean().item(),
            "arm2_dense": r_arm2.mean().item(),
            "settle": r_settle.mean().item(),
            "arm1_home": r_arm1_home.mean().item(),
            "joint_limits": r_limits.mean().item(),
            "posture": r_posture.mean().item(),
            "contact": r_contact.mean().item(),
            "action_rate": r_rate.mean().item(),
            "speed1": speed1.mean().item(),
            "speed2": speed2.mean().item(),
            "dev1_from_home": dev1.mean().item(),
            "dev2_from_home": dev2.mean().item(),
            "dist_ee1_obj": dist_ee1_obj.mean().item(),
            "dist_ee1_give": dist_ee1_give.mean().item(),
            "dist_ee2_recv": dist_ee2_recv.mean().item(),
            "dist_obj_handoff": dist_obj_handoff_xy.mean().item(),
            "start_in_hand_prob": self._start_in_hand_prob(),
        }

        return reward

    # ------------------------------------------------------------------ #
    #  Termination                                                       #
    # ------------------------------------------------------------------ #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_z = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        terminated = obj_z < self.cfg.drop_height  # fell below the table top
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # We deliberately do NOT terminate on success. Letting the episode keep running
        # is what makes s6's 4.0/step annuity beat parking at s4, without having to tune
        # a terminal bonus against the forfeited remaining reward.
        return terminated, truncated

    # ------------------------------------------------------------------ #
    #  Optional start-in-hand curriculum (off by default)                #
    # ------------------------------------------------------------------ #
    def _start_in_hand_prob(self) -> float:
        """Fraction of resets that start already grasped + lifted.

        Anneals linearly from curriculum_start_prob -> 0 over curriculum_end_step env
        steps. Disabled unless cfg.use_start_in_hand is True — it turned out to be
        unnecessary once the geometry was fixed, so flat training is the default.
        """
        if not getattr(self.cfg, "use_start_in_hand", False):
            return 0.0
        step = float(getattr(self, "common_step_counter", 0))
        frac = step / max(1.0, float(self.cfg.curriculum_end_step))
        return max(0.0, self.cfg.curriculum_start_prob * (1.0 - frac))

    # ------------------------------------------------------------------ #
    #  Reset                                                             #
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids) -> None:
        # (1) base bookkeeping — resets episode_length_buf etc.
        super()._reset_idx(env_ids)

        # (2) both arms back to the ready pose
        for robot in (self.robot_1, self.robot_2):
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # (3) reset the object with a little x,y jitter (env-local -> add origins)
        root_state = self.object.data.default_root_state[env_ids].clone()
        n = len(env_ids)
        rand_xy = torch.zeros((n, 2), device=self.device).uniform_(
            -self.cfg.reset_object_xy_range, self.cfg.reset_object_xy_range)
        root_state[:, 0:2] += rand_xy
        root_state[:, 0:3] += self.scene.env_origins[env_ids]  # local -> world

        # (4) clear grasp + offset state
        self._held_by_1[env_ids] = False
        self._held_by_2[env_ids] = False
        self._prev_held_by_1[env_ids] = False
        self._prev_held_by_2[env_ids] = False
        self._off_p1[env_ids] = 0.0
        self._off_p2[env_ids] = 0.0
        self._off_q1[env_ids] = 0.0
        self._off_q2[env_ids] = 0.0
        self._off_q1[env_ids, 0] = 1.0
        self._off_q2[env_ids, 0] = 1.0

        # (4b) optional curriculum: start a fraction of envs already holding + lifted.
        # We set held_1=True with an identity offset and lift the object; the first weld
        # substep snaps it onto arm1's EE wherever the arm settles. (We can't read the
        # object pose reliably right after write_joint_state without a sim step, so we
        # lean on the weld rather than reading EE pose here.) Off unless use_start_in_hand.
        p = self._start_in_hand_prob()
        if p > 0.0 and n > 0:
            in_hand = torch.rand(n, device=self.device) < p
            if in_hand.any():
                local_ih = in_hand.nonzero(as_tuple=False).squeeze(-1)
                ih_ids = env_ids[local_ih]
                root_state[local_ih, 2] = (
                    self.scene.env_origins[ih_ids, 2] + self._table_top_z + 0.15
                )
                self._held_by_1[ih_ids] = True  # identity offset already set above

        self.object.write_root_state_to_sim(root_state, env_ids=env_ids)

        # (5) reset the best-distance buffers (fresh well each episode)
        self._best_d_ee1_give[env_ids] = 999.0
        self._best_d_obj_handoff[env_ids] = 999.0

        # (6) reset action-rate history for these envs
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0.0