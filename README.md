# Long Horizon Mobile Manipulation Upon HoMeR

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](https://homer-manip.github.io/assets/paper.pdf)  
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](https://homer-manip.github.io)

---

## Overview

We present a long horizon mobile manipulation framework built upon **HoMeR** (Hybrid Whole-Body Policies for Mobile Robots) as an a hybrid imitation learning framework for mobile manipulation. It combines high level Gemini / GPT4o planner for single task policy chaining. With LAMA segmentation and inpainting, single task policies can be combined for generalized skill variants unseen without requiring fine tuning of a large generalist model.

Automated KISA task parsing and keyframe + keypoint labeling with whole-body control with a hybrid action representation achieves generalizable and precise robot behavior in both simulation and real-world settings.

<table>
  <tr>
    <td><img src="readme_assets/pillow.gif" width="250"/></td>
    <td><img src="readme_assets/remote.gif" width="250"/></td>
    <td><img src="readme_assets/sweeping.gif" width="250"/></td>
  </tr>
  <tr>
    <td><img src="readme_assets/cube.gif" width="250"/></td>
    <td><img src="readme_assets/dishwasher.gif" width="250"/></td>
    <td><img src="readme_assets/cabinet.gif" width="250"/></td>
  </tr>
</table>



---

### 🖥️ Simulation-Only

📄 [`SIM.md`](SIM.md)

This guide covers:
- Conda setup on macOS and Linux
- Simulated data collection and annotation
- Training and evaluating HoMeR and baselines in simulation

---

### 🤖 Real-World

📄 [`REAL.md`](REAL.md)

This guide covers:
- Hardware and software setup for real-world deployment
- Real-world data collection and annotation
- Training and evaluating HoMeR and baselines in real

---

## Repository Structure

```bash
cfgs/                 # Training config files
envs/                 # Environment setup for sim and real
docker/               # Real-world Docker setup
scripts/              # Training and evaluation scripts
interactive_scripts/  # Data collection, replay, and data annotation tools
dataset_utils/        # Dataset loading and data visualization tools
mj_assets/            # MJCF assets for simulation
sbatch_scripts/       # SLURM scripts to launch training jobs

