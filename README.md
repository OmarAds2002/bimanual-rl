# Bimanual UR10 Cylinder Handoff (Isaac Lab)

Two UR10 arms face each other across a table and learn to hand a cylinder between them.
Arm 1 picks the object up, carries it to the middle, and passes it to arm 2. The object
starts far enough onto arm 1's side that arm 2 **cannot reach it** — so the handoff has to
happen, it isn't just one of several ways to solve the task.

Built as a `DirectRLEnv` in Isaac Lab and trained with `rsl_rl` PPO across 1024 parallel
environments.

![handoff demo](docs/demo.gif)

## Results

| Metric | Value |
|---|---|
| Success rate (handover complete) | **0.84** |
| Drop rate | **0.00** — see the note below |
| Post-handoff behavior | arms settle to a stable hold |
| Throughput | ~19.6k steps/s @ 1024 envs (RTX 5080) |
| Convergence | ~iteration 700–2500 |
| Curriculum | none needed — flat training |

Trained on: i9-14900KF / RTX 5080 16GB / Ubuntu 24.04.

> **About that drop rate.** 0.00 looks better than it is. The grasp in this project is a
> kinematic weld, not real contact — there's no friction and no force closure, so there's
> almost no physical way for the object to fall. The number mostly tells you the task
> can't produce drops, not that the policy is careful. It would start to mean something
> with a real gripper.

> **About the success rate.** 0.84 is from a single training run. Deep RL results move
> between random seeds, so treat this as one sample rather than a settled number. Running
> multiple seeds and reporting a range is on the list.

## The task in more detail

**Plain UR10, no gripper.** This Isaac Lab build ships the UR10 without a Robotiq gripper,
so a "grasp" here is a kinematic weld rather than fingers closing on the object. It latches
when the end-effector is close enough and the policy asks for a grasp, then holds the object
at whatever relative position it had at that moment. **This is not physically realistic.** A
real grasp involves finger contact, friction, and force closure, and none of that is modeled.
That's a deliberate choice about scope: the point of this project is the *handoff behavior* —
approach, timing, both arms holding at once, release — not contact-rich manipulation. Adding
a real gripper is the main planned next step.

**Long object.** The cylinder is a 0.38 m baton with a give point at one end and a receive
point at the other. Arm 1 takes the body, arm 2 aims for the far end, so the two wrists
aren't competing for the same 10 cm of space.

**Action space (14):** 6 joint targets per arm, plus 2 grasp-intent values.

**Observation space (51):**
- 24 — joint positions and velocities, both arms
- 13 — object pose and velocity
- 6 — end-effector → object vectors
- 6 — arm → grasp-target vectors (arm 1 to the give end, arm 2 to the receive end)
- 2 — grasp flags

**Reward.** A ladder of stages — reach, lift, carry, arrive, *both arms holding*, handover,
settle — where each stage pays a higher rate than the one below it, plus some bounded dense
shaping to guide movement within a stage. The "both arms holding" rung is the important one:
the receiver has to have the object before the giver lets go, which means completing the
handoff is never worse than stopping partway. The final settle stage brings the arms to rest
after the transfer instead of leaving them drifting.

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
- `ffmpeg` — only if you want the in-script video recording

Versions this was built and tested against:

| Package | Version |
|---|---|
| isaacsim | 5.1.0 |
| torch | 2.7.0 (cu128) |
| torchvision | 0.22.0 (cu128) |
| python | 3.11 |

## Setup

### 1. Isaac Sim + Isaac Lab

Follow the official pip install for Isaac Lab v2.3.2:
<https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html>

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

> ⚠️ **Once this env works, don't run pip upgrades.** No `pip install -U`, no
> `pip install --upgrade`, and no install that resolves and pulls new dependencies. Isaac
> Sim pins very specific `torch` and `numpy` builds, and in my experience any upgrade
> quietly swaps them out and breaks the whole env — import errors, CUDA mismatches. If you
> really need another package, install it with `--no-deps` and check nothing else moved.
> If it's already broken, rebuilding the conda env from scratch is faster than trying to
> undo the upgrade.

### 2. This project

From the repo root, with `env_isaaclab` active:

```bash
cd bimanual-rl
python -m pip install -e source/bimanual_handoff
```

If that tries to pull a different `torch` or `numpy`, stop it and re-run with `--no-deps`
(see the warning above), then install any genuinely missing small dependencies by hand.

## Usage

All scripts launch through Isaac Lab's Python so the Kit runtime resolves. `isaaclab.sh`
lives in your Isaac Lab clone (`~/IsaacLab` above), not in this repo.

### Train

```bash
cd bimanual-rl
rm -rf logs/p3_handoff/     # ALWAYS clear this if you changed the reward or physics
~/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 1024 --headless
```

Checkpoints are written every 100 iterations to `logs/p3_handoff/model_*.pt`. Ctrl-C is safe.

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

Runs headless with cameras enabled and captures the first `--video_length` frames. Needs
`ffmpeg` on your PATH.

**Option B — screen record.** Run `play.py` normally, position the viewport camera, and
capture the window (OBS, SimpleScreenRecorder). For a portfolio clip this often looks
better since you control the framing.

To turn an mp4 into a GIF for this README:

```bash
ffmpeg -i videos/play/handoff-step-0.mp4 \
  -vf "fps=15,scale=720:-1:flags=lanczos" docs/demo.gif
```

## Problems worth writing down

A few things that were harder than they looked.

### The bug that looked like a reward problem

For a long time, arm 2 would approach the object and then drift off into an endless orbit,
and no amount of reward tuning fixed it. The real problem was geometric. The grasp weld
snapped the object's center onto the holder's end-effector, so "arm 2 reaches the object"
actually meant "two UR10 wrists try to occupy the same 10 cm" — which is two solid bodies
overlapping, not a grasp. The plateau I kept hitting at a suspiciously exact distance
wasn't the reward being too sparse, it was the arms physically running into each other.

The fix was to record the object's position *relative to* the end-effector at the moment of
grasp, and to split the object into two separate grasp points.

**Lesson:** if a plateau sits at an oddly specific distance, check the geometry before
touching the reward.

### Sitting still beat finishing, and the value network was right

Even after the geometry fix, the policy preferred to park at the "arrived" stage instead of
attempting the transfer. The reason was arithmetic.

Sitting still earned about 1 point per step for the rest of the episode. Over a few hundred
steps that adds up to roughly 100. Completing the handoff paid a one-time bonus of 2, risked
a drop penalty, and — the real killer — **ended the episode**, deleting all the reward that
would have come after it. So the value network learned that finishing the task was about a
98-point mistake. It wasn't wrong. My reward was.

Two fixes:
1. **Don't end the episode on success.** Add a settle stage afterwards so finishing also
   earns an ongoing stream, and the comparison becomes stream-vs-stream instead of
   stream-vs-lump-sum.
2. **Make later stages pay a higher rate**, not a bigger one-off bonus.

**Lesson:** when a policy refuses to do the obvious thing, add up what each option is
actually worth before assuming the policy is failing to learn.

### A silent overwrite in the reward code

The "settled" stage never fired. The cause: the settled mask is a subset of the "handover
complete" mask, and the code assigned the handover reward *after* the settle reward — so it
overwrote it every single step, silently.

When you assign rewards using boolean masks, overlapping masks are a failure that produces
no error message, and the order of assignment matters. Fix: make the stages mutually
exclusive before assigning anything.

### A pose that was free because physics wasn't computing it

Arm 2 kept folding its wrist through its own upper arm. The stock UR10 config ships with
self-collisions turned off, so PhysX never calculated those contacts — the folded pose was
literally free to enter. No reward term can penalize a collision the simulator isn't
computing.

Fix: turn on self-collisions for the articulation, and clamp joint targets to the soft
limits so the arm stops slamming into its stops.

## Known limitations

- **The grasp is a kinematic weld, not real contact.** No fingers, friction, or force
  closure. Swapping in a real gripper is the main next step — and it would break things:
  the policy has never been penalized for approach speed (the weld latches at any velocity,
  real fingers would swat the object away), and having both arms clamped on one rigid baton
  would create real internal forces that this reward doesn't model at all.
- **No force or contact sensing in the observation.** The policy has no way to know it's
  actually holding something. A wrist force-torque sensor would be the first addition.
- **Single seed.** Results are from one run.
- Contact sensors are stubbed out (a guarded no-op); the self-collision penalty only
  activates if they're wired up.

## References

- robosuite `TwoArmHandover` — the both-arms-holding stage, splitting the object into two
  grasp points, bounded tanh shaping
- Bi-DexHands `ShadowHandOver`
- Ng, Harada & Russell 1999 — reward shaping that provably doesn't change the optimal
  policy. Worth noting: the stage ladder here does **not** satisfy their condition, since
  it pays for *staying* in a stage rather than for *moving between* stages. The strict
  ordering of the rungs is a hand-enforced substitute for their guarantee.
- Florensa et al. 2017 — reverse curriculum generation (tried, not used in the end)
