# Bimanual UR10 Cylinder Handoff (Isaac Lab)

Two opposed UR10 arms learn to hand a cylinder across a shared table with PPO. Arm 1
(the giver) picks the object off the table, carries it to a central handoff point, and
transfers it to arm 2 (the receiver). The object starts deep on arm 1's side so arm 2
**physically can't** reach it directly — the handoff is forced, not optional.

Built as a `DirectRLEnv` in Isaac Lab, trained with `rsl_rl` PPO across 1024 parallel
environments.

<!-- Replace with your recording. See "Recording a video" below for how to make one.
     For inline autoplay on GitHub, commit a GIF; for an mp4, drag-and-drop it into the
     GitHub README editor to get an embeddable link. -->
![handoff demo](docs/demo.gif)

## Results

| Metric | Value |
|---|---|
| Success rate (handover complete) | **0.84** |
| Drop rate | **0.00** |
| Post-handoff behavior | arms settle to a stable hold |
| Throughput | ~19.6k steps/s @ 1024 envs (RTX 5080) |
| Convergence | ~iteration 700–2500 |
| Curriculum | none needed — flat training |

Trained on: i9-14900KF / RTX 5080 16GB / Ubuntu 24.04.

## The task in a bit more detail

- **Plain UR10, no gripper.** This Isaac Lab build ships the UR10 without a Robotiq
  gripper, so a "grasp" is a kinematic weld rather than finger contact. It latches
  when the end-effector is in range and the policy commands a grasp, and holds the
  object at the relative transform captured at contact. **This is not physically
  accurate** — a real grasp involves finger contact, friction, and force closure, none
  of which the weld models. That's a deliberate scope choice: the focus of this project
  is the *handoff behavior* (approach, transfer timing, mutual grasp, release), not
  contact-rich manipulation. Adding a real gripper (Robotiq, contact-based grasping) is
  planned future work once time allows.
- **Elongated object.** The cylinder is a 0.38 m baton with give/receive sites at each
  end. Arm 1 grabs the body, arm 2 aims for the far end, so the two wrists never fight
  for the same 10 cm of space.
- **Action space (14):** 6 joint targets per arm + 2 grasp-intent scalars.
- **Observation space (51):** 24 joint state (pos+vel, both arms), 13 object state
  (pose + velocity), 6 end-effector→object vectors, 6 arm→grasp-target vectors (arm1 to
  the give end, arm2 to the receive end), 2 grasp flags.
- **Reward:** a monotone stage ladder (reach → lift → carry → arrive → *mutual grasp* →
  handover → settle) plus bounded dense shaping. The mutual-grasp rung is the key piece
  — both arms hold at once before the giver lets go, so completing the handoff is never
  worse than just parking. The final settle stage brings the arms to a stable rest after
  the transfer instead of leaving them drifting.

## Repository structure

```
bimanual-rl/
├── scripts/
│   ├── train.py                    # PPO training entry point
│   └── play.py                     # load a checkpoint, watch or record
├── source/bimanual_handoff/
│   └── bimanual_handoff/tasks/handoff/
│       ├── handoff_env.py          # the DirectRLEnv
│       ├── handoff_env_cfg.py      # env / scene / task config
│       ├── __init__.py             # gym.register("Isaac-Handoff-Direct-v0")
│       └── agents/
│           └── rsl_rl_ppo_cfg.py   # PPO hyperparameters
├── pyproject.toml
└── README.md
```

## Requirements

- Ubuntu 22.04+ (GLIBC ≥ 2.35 — Ubuntu 24.04 is fine)
- NVIDIA GPU with a recent driver (developed on an RTX 5080)
- **Python 3.11** (required by Isaac Sim 5.x)
- Isaac Sim **5.1.0** + Isaac Lab **v2.3.2**
- `rsl_rl` (installed with Isaac Lab)
- `ffmpeg` — only needed if you want the in-script video recording

Pinned versions this was built and tested against:

| Package | Version |
|---|---|
| isaacsim | 5.1.0 |
| torch | 2.7.0 (cu128) |
| torchvision | 0.22.0 (cu128) |
| python | 3.11 |

## Setup

### 1. Isaac Sim + Isaac Lab

Follow the official pip install for Isaac Lab v2.3.2:
https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html

The short version (Linux, conda):

```bash
# conda env on Python 3.11
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install --upgrade pip                       # upgrading pip ITSELF is fine

# Isaac Sim 5.1.0 + a matching CUDA torch build
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab (from source), with rsl_rl
sudo apt install cmake build-essential
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
./isaaclab.sh --install rsl_rl                   # or "--install" for all frameworks
```

> ⚠️ **Once this env works, do not run pip upgrades — period.** No `pip install -U`, no
> `pip install --upgrade`, and no package install that resolves and pulls new deps. Isaac
> Sim pins very specific `torch` / `numpy` / dependency builds, and in my experience any
> upgrade silently swaps them out and breaks the whole conda env every time (import
> errors, CUDA mismatches). If you genuinely need another package, install it with
> `--no-deps` and check that nothing else moved — otherwise leave the env alone. If it's
> already broken, the reliable fix is to rebuild the conda env from scratch rather than
> try to un-upgrade.

### 2. This project

From the repo root, with the `env_isaaclab` conda env active:

```bash
cd bimanual-rl
python -m pip install -e source/bimanual_handoff
```

If that install tries to pull a different `torch`/`numpy`, stop it and re-run with
`--no-deps` (see the warning above), then install any genuinely missing small deps by
hand.

## Usage

All scripts launch through Isaac Lab's Python so the Kit runtime resolves. `isaaclab.sh`
lives in your Isaac Lab clone (`~/IsaacLab` above), not in this repo.

### Train

```bash
cd bimanual-rl
rm -rf logs/p3_handoff/     # ALWAYS clear this if you changed the reward or physics
~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 1024 --headless
```

Checkpoints are written every 100 iterations to `logs/p3_handoff/model_*.pt`.
Ctrl-C is safe.

### Watch a trained policy

```bash
# loads the latest checkpoint automatically
~/IsaacLab/isaaclab.sh -p scripts/play.py --num_envs 1

# or a specific one
~/IsaacLab/isaaclab.sh -p scripts/play.py --checkpoint logs/p3_handoff/model_2500.pt
```

### Monitor training

```bash
tensorboard --logdir logs/p3_handoff
```

Watch `success_frac` (any completed handover) and `drop_frac`. The per-stage fractions
(`stage_1_frac` … `stage_7_frac`) show where the policy is spending its time.

## Recording a video

**Option A — in-script (reproducible).** `play.py` has a `--video` flag that renders
offscreen and writes an mp4 to `videos/play/`:

```bash
~/IsaacLab/isaaclab.sh -p scripts/play.py --num_envs 1 --video --video_length 600
```

This runs headless with cameras enabled and captures the first `--video_length` frames.
Needs `ffmpeg` on your PATH.

**Option B — screen record.** Run `play.py` normally, position the viewport camera at a
good angle, and capture the window (OBS, SimpleScreenRecorder, etc.). For a portfolio
clip this is often the nicer-looking option since you control the framing.

To turn an mp4 into a GIF for inline display in this README:

```bash
ffmpeg -i videos/play/handoff-step-0.mp4 \
  -vf "fps=15,scale=720:-1:flags=lanczos" docs/demo.gif
```

## Notable engineering challenges

A few problems that were more interesting than they first looked:

**The bug that looked like a reward problem.** For a long stretch, arm 2 would approach
the object and then veer off in an endless orbit, and no amount of reward tuning fixed
it. The actual defect was geometric: the grasp weld snapped the object's center onto the
holder's end-effector origin, so "arm 2 reaches the object" really meant "two UR10
wrists occupy the same 10 cm" — which is interpenetration, not a grasp. The plateau I
kept hitting at a suspiciously specific distance wasn't a dry reward well, it was the
collision manifold. Fixing it meant capturing the object's pose *relative* to the
end-effector at grasp time and splitting the object into two grasp sites.

**Parking beat completing, and the critic was right.** Even after the geometry fix, the
policy preferred to sit at the "arrived" stage rather than attempt the transfer. Doing
the math: parking earned a safe ~1.0/step for the rest of the episode, while attempting
the handoff risked a drop penalty *and* termination. A 2× per-step bonus for completing
wasn't nearly enough to beat that annuity. The fix was a monotone ladder with a
mutual-grasp rung and a 4× completion bonus, plus not terminating on success so the
annuity comparison stays honest.

**A silent boolean-mask overwrite.** The "settled" reward stage never fired. The cause:
the settled mask is a subset of the "handover complete" mask, and the code assigned the
handover reward *after* the settle reward — silently clobbering it every step. With
boolean-mask reward assignment, overlapping masks are a silent failure and assignment
order is load-bearing. Fix: make the stages disjoint before assigning anything.

**A pose that was free because physics wasn't computing it.** Arm 2 kept folding its
wrist through its own upper arm. The stock UR10 config ships with self-collisions
disabled, so PhysX never computed those contacts — the folded configuration was
literally free to enter. No reward term can penalize a collision the simulator isn't
calculating. Fix: enable self-collisions on the articulation (and clamp joint targets to
the soft limits so the arm stops slamming into its stops).

## Known limitations

- **The grasp is a kinematic weld, not real contact** (see the task description above).
  No fingers, friction, or force closure are modeled. Swapping in a real gripper is the
  main piece of planned future work.
- Contact sensors are stubbed (a guarded no-op); the self-collision penalty term only
  activates if they're wired.

## References

- robosuite `TwoArmHandover` — mutual-grasp rung, handle/head split, bounded tanh shaping
- Bi-DexHands `ShadowHandOver`
- Ng et al. 1999 — potential-based reward shaping
- Florensa et al. 2017 — reverse curriculum generation (explored, ultimately unused here)