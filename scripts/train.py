# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""P3 Bimanual Handoff — PPO training.

    ~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 1024 --headless
    ~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 256     # with a viewer

If you changed the reward or the physics, clear the old run first:
    rm -rf logs/p3_handoff/
otherwise you can end up training on top of a stale checkpoint.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train P3 Bimanual Handoff")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

# Boot Kit before the isaac imports (same reason as play.py).
app = AppLauncher(headless=args.headless).app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import bimanual_handoff.tasks.handoff  # noqa: F401  (registers the env)
from bimanual_handoff.tasks.handoff.handoff_env_cfg import HandoffEnvCfg
from bimanual_handoff.tasks.handoff.agents.rsl_rl_ppo_cfg import HandoffPPORunnerCfg

# --- env ---
env_cfg = HandoffEnvCfg()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-Handoff-Direct-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)  # rsl_rl expects a VecEnv interface

# --- runner + train ---
runner_cfg = HandoffPPORunnerCfg()
runner_cfg.device = "cuda:0"

runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir="logs/p3_handoff", device="cuda:0")
runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

# --- cleanup ---
env.close()
app.close()