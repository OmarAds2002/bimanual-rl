# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""P3 Bimanual Handoff — play a trained policy.

Watch it live:
    ~/IsaacLab/isaaclab.sh -p scripts/play.py --checkpoint logs/p3_handoff/model_1000.pt
    ~/IsaacLab/isaaclab.sh -p scripts/play.py            # loads the latest checkpoint

Record an mp4 instead (renders offscreen, writes to videos/play/):
    ~/IsaacLab/isaaclab.sh -p scripts/play.py --video --video_length 600
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play P3 Bimanual Handoff")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--video", action="store_true", help="Record an mp4 of the rollout")
parser.add_argument("--video_length", type=int, default=600, help="Frames to record for --video")
args = parser.parse_args()

# Recording renders offscreen, so it needs cameras enabled and runs headless. If
# you just want to watch, leave --video off and a viewer window opens.
app = AppLauncher(
    headless=args.headless or args.video,
    enable_cameras=args.video,
).app

# Heavy imports go AFTER the app boots — isaac/pxr/carb only resolve once Kit is up.
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import bimanual_handoff.tasks.handoff  # noqa: F401  (registers the env)
from bimanual_handoff.tasks.handoff.handoff_env_cfg import HandoffEnvCfg
from bimanual_handoff.tasks.handoff.agents.rsl_rl_ppo_cfg import HandoffPPORunnerCfg

# --- env ---
env_cfg = HandoffEnvCfg()
env_cfg.scene.num_envs = args.num_envs

# render_mode="rgb_array" is what lets RecordVideo grab frames.
render_mode = "rgb_array" if args.video else None
env = gym.make("Isaac-Handoff-Direct-v0", cfg=env_cfg, render_mode=render_mode)

# Wrap with RecordVideo BEFORE the rsl_rl wrapper. Records from step 0 for
# video_length frames, then env.close() flushes the file.
if args.video:
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos/play",
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        name_prefix="handoff",
        disable_logger=True,
    )

env = RslRlVecEnvWrapper(env)

# --- runner + checkpoint ---
runner_cfg = HandoffPPORunnerCfg()
runner_cfg.device = "cuda:0"

log_dir = "logs/p3_handoff"
runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

# Explicit path if given, else the highest-numbered model_*.pt in log_dir.
if args.checkpoint:
    ckpt_path = args.checkpoint
else:
    ckpts = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")],
        key=lambda f: int(f.replace("model_", "").replace(".pt", "")),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    ckpt_path = os.path.join(log_dir, ckpts[-1])

print(f"Loading checkpoint: {ckpt_path}")
runner.load(ckpt_path)

# get_inference_policy is the right handle in this rsl_rl version — there's no
# runner.alg.actor_critic to reach into.
policy = runner.get_inference_policy(device="cuda:0")

# env.get_observations() returns just the obs here, NOT an (obs, extras) tuple.
obs = env.get_observations()

if args.video:
    print(f"Recording {args.video_length} frames to videos/play/ ...")
else:
    print("Playing... close the viewer to exit.\n")

step = 0
while True:
    with torch.no_grad():
        actions = policy(obs)
    obs, _, _, _ = env.step(actions)
    step += 1
    if args.video and step >= args.video_length:
        break

env.close()   # flushes the mp4 when recording
app.close()