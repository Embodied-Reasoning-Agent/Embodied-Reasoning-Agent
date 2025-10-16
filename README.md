<h1 align="center">
  <!-- <img src="docs/images/embodied-logo.png" alt="era-logo" width="40" height="40" style="vertical-align: middle; margin-top: -12px;"> -->
  ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online Reinforcement Learning
</h1>

<p align="center">
  <!-- 📄 <a href="http://arxiv.org/abs/2502.xxxxx"><strong>Paper</strong></a> | -->
  🤗 <a href="https://huggingface.co/EmbodiedReasoningAgent"><strong>Dataset</strong></a> |
  🏠 <a href="https://embodied-reasoning-agent.github.io"><strong>Project Website</strong></a>
</p>

<p align="center">
  <a href="">Hanyang Chen*</a>,
  <a href="">Mark Zhao*</a>,
  <a href="">Rui Yang*</a>,
  <a href="">Qinwei Ma</a>,
  <a href="">Ke Yang</a>,
  <a href="">Jiarui Yao</a>,
  <a href="">Kangrui Wang</a>,
  <a href="">Hao Bai</a>,
  <a href="">Zhenhailong Wang</a>,
  <a href="">Rui Pan</a>,
  <a href="">Mengchao Zhang</a>,
  <a href="">Jose Barreiros</a>,
  <a href="">Aykut Onol</a>,
  <a href="">ChengXiang Zhai</a>,
  <a href="">Heng Ji</a>,
  <a href="">Manling Li</a>,
  <a href="">Huan Zhang</a>,
  <a href="">Tong Zhang</a>
</p>

<p align="center">
  <sup>1</sup>University of Illinois Urbana-Champaign, 
  <sup>2</sup>Northwestern University, 
  <sup>3</sup>Toyota Research Institute
</p>

<p align="center">
  <img src="docs/images/framework.png" width="100%" alt="ERA Framework">
</p>

# 🔥 Overview

We introduce the  **Embodied Reasoning Agent (ERA)** , a framework that transforms a compact **Vision Language Model (VLM)** into a performant and efficient embodied agent. In this work, we study: **1. What prior knowledge does embodied agent require before RL** and **2 What make RL in long-horizon embodied task stable and effective?** We distill them into a **unified post-training regime** that is capable of delivering **both** high-level planning agent and low-level control agent, by different curation of training data.

ERA takes a **two-stage approach.** Stage 1: Infuse **foundational capabilities** into the model, which is categorized into  **three kinds of embodied prior knowledge** , and Stage 2: refine the agent by **online reinforcement learning** with **rich process reward** and  **turn-level GAE** .

# 📋 Table of Contents

- [🎓 Stage 1: Embodied Prior Learning (EPL)](#-embodied-prior-learning-epl)
  - [EPL Setup](#epl-setup)
  - [EPL Dataset Preparation](#epl-dataset-preparation)
  - [EPL Training](#epl-training)
- [🚀 Stage 2: Reinforcement Learning (RL)](#-reinforcement-learning-rl)
  - [RL Setup](#rl-setup)
  - [RL Dataset Preparation](#rl-dataset-preparation)
  - [RL Training](#rl-training)

---

# 🎓 Stage 1: Embodied Prior Learning (EPL)

## EPL Setup

Download repo:

```bash
git clone git@github.com:Embodied-Reasoning-Agent/Embodied-Reasoning-Agent.git
cd Embodied-Reasoning-Agent
```

Environment setup:

```bash
cd ERA-sft
conda env create -f environment.yaml 
conda activate era-epl-env
```

## EPL Dataset Preparation

1. **Environment-Anchored Prior Dataset**

   - Download the dataset from [EB-Man_environment_anchored_prior_dataset](https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-Man_environment_anchored_prior_dataset) and [EB-ALFRED_environment_anchored_prior_dataset](https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-ALFRED_environment_anchored_prior_dataset)
2. **External Knowledge Prior Dataset**

   - Download the dataset from [EB-Man_external_knowledge_prior_dataset](https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-Man_external_knowledge_prior_dataset) and [EB-ALFRED_external_knowledge_prior_dataset](https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-ALFRED_external_knowledge_prior_dataset)

Note: Place either the environment-anchored prior data or the external knowledge prior data according to the structure defined in [`ERA-sft/epl/data/stage1.yaml`](./ERA-sft/epl/data/stage1.yaml)

3. **Trajectory-Augmented Prior Dataset**
   - Download the dataset from [EB-Man_trajectory_augmented_prior_dataset](https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-Man_trajectory_augmented_prior_dataset) and [EB-ALFRED_trajectory_augmented_prior_dataset](https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-ALFRED_trajectory_augmented_prior_dataset)
   - Place the data according to the structure defined in [`ERA-sft/epl/data/stage2.yaml`](./ERA-sft/epl/data/stage2.yaml)

## EPL Training

1. Configure your training settings:

```bash
cd epl
```

- Open `scripts/train.sh`
- Set the `SFT_TASK` variable to specify your training stage
- Set the `SAVE_DIR` variable to specify path to your saving directory
- Set the `IMAGE_FOLDER` variable to specify path to your image folder

2. Start training:

```bash
bash scripts/train.sh
```

---

# 🚀 Stage 2: Online Reinforcement Learning (RL)

## Online RL Overview

Note that since we are working with online reinforcement learning, so there are inevitably two parts of workflow: **1. The RL Training Algorithm** and **2. The RL Environment**. We have seperate **code/environment** for these two parts. Another important thing is, we adopt **Parrallel Rollout** in this framework, implemented in a server-client manner, and we will talk about it in [RL Training](#rl-training). Now, we will first talk about setup the running environment for these two parts.

### 1. Setting up the environment for RL Training Algorithm

Install the RL training framework in a virtual environment:

```bash
cd ERA-rl
python -m venv era-rl-env
source era-rl-env/bin/activate  # On Windows: era-rl-env\Scripts\activate
pip install -r requirement.txt
deactivate  # you can now de-activate the env to setup other env
```

### 2. Setting up the environment for RL Environment

We support two embodied environments from EmbodiedBench: **EB-ALFRED** and **EB-Manipulation**. Follow the instructions below to set up each environment.

> Note that this line: ```cd ERA-rl/VAGEN/vagen/env/Embench_new``` in the following is very important, since the codebase is rather nested, it keeps you in the right place for installation.

#### 2.1. EB-ALFRED Environment

**2.1.1 Environment Installation**

```bash
cd ERA-rl/VAGEN/vagen/env/Embench_new
conda env create -f conda_envs/environment.yaml
conda activate embench
pip install -e .
```

**2.1.2 Additional Installation**

Download dataset from huggingface.
```bash
conda activate embench
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```
Run the following code to ensure the EB-ALFRED environment is working correctly. `Remember to start headless server.`

```bash
conda activate embench
python -m embodiedbench.envs.eb_alfred.EBAlfEnv
```

#### 2.2. EB-Manipulation Environment

**2.2.1 Environment Installation**

```bash
cd ERA-rl/VAGEN/vagen/env/Embench_new
conda env create -f conda_envs/environment.yaml
conda activate embench
pip install -e .
```

**2.2.2 Additional Installation**

* Install Coppelia Simulator

CoppeliaSim V4.1.0 required for Ubuntu 20.04; you can find other versions here (https://www.coppeliarobotics.com/previousVersions#)

```bash
conda activate embench_man
cd embodiedbench/envs/eb_manipulation
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/ /PATH/YOU/WANT/TO/PLACE/COPPELIASIM
```

* Add the following to your *~/.bashrc* file:

```bash
export COPPELIASIM_ROOT=/PATH/YOU/WANT/TO/PLACE/COPPELIASIM
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

> Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

* Install the PyRep, EB-Manipulation package and dataset:
```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
pip install -r requirements.txt
pip install -e .
cp ./simAddOnScript_PyRep.lua $COPPELIASIM_ROOT
git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation
mv EB-Manipulation/data/ ./
rm -rf EB-Manipulation/
cd ../../..
```

> Remember that whenever you re-install the PyRep, simAddOnScript_PyRep.lua will be overwritten. Then, you should copy this again.

* Run the following code to ensure the EB-Manipulation is working correctly (start headless server if you have not):
```bash
conda activate embench_man
export DISPLAY=:1
python -m embodiedbench.envs.eb_manipulation.EBManEnv
```

---

**Congratulations, you have finished the environment setup for online RL, and the following would be very easy to go.**

## RL Dataset Preparation

Since the dataset preparation in online, simulator-based RL basically can be translated to the initial state(or task ID) to start with(mostly heavy-lifting is done by the simulator), we just have to specify the sequence of task ID here.



## RL Training

Coming soon...
