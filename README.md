# Hyperparameter Tuning for DQN, PPO, and SAC

This repository provides instructions for hyperparameter tuning in DQN, PPO, and SAC after modifications were made to improve PPO and DQN as the game complexity increased. Notably, some changes have been implemented to enhance PPO's performance.

## Setup Instructions

### 1. Load Required Modules on Compute Canada (CC)

Before running experiments, ensure that the required modules are loaded. Run the following commands:

```bash
module load clang/17.0.6
module load cmake
module load python/3.11.5
module load scipy-stack
module load glfw
module load cuda/12.2
module load opencv/4.11.0
```

### 2. Install Required Packages

Install the necessary dependencies:

```bash
pip install --no-index wandb
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/vedudx/pfrl.git@gymnasium_support
pip install imageio
```

### 3. Install AgarLE

Navigate to your AgarLE directory and install the package:

```bash
cd /home/mayman/projects/def-machado/mayman/AgarLE
python setup.py install --user
```

### 4. Login to Weights & Biases (wandb)

Run the following command to log in to wandb:

```bash
wandb login
```

> **Note:** All the above installations are one-time processes and do not need to be repeated for every experiment.

## Running Hyperparameter Tuning

A script has been implemented to automate the hyperparameter tuning process, including submitting jobs to Compute Canada.

### DQN Hyperparameter Tuning

Run the following command to generate and submit the DQN tuning script:

```bash
python generate_bash.py DQN.sh /home/mayman/projects/def-machado/mayman/agarle_bench/AgarLE-benchmark/DQN_full_action_set.py \
  --lr=1e-5,3e-5,1e-4,3e-4 \
  --batch_accumulator="sum","mean" \
  --minibatch_size=32,64 \
  --tau=1e-2,1e-3,5e-3 \
  --epochs=1,2,5
```

### SAC Hyperparameter Tuning

Run the following command for SAC tuning:

```bash
python generate_bash.py SAC.sh SAC_full_action_set.py \
  --reward="reward_gym","min_max" \
  --lr=3e-5,1e-4,1e-5 \
  --soft-update-tau=0.01,0.005,0.001 \
  --max-grad-norm=0.5,0.7,0.9
```

### PPO Hyperparameter Tuning

PPO requires a two-step tuning process due to the number of hyperparameters involved:

#### Step 1: Initial Run

Run the following command to determine initial best parameters:

```bash
python generate_bash.py PPO.sh /home/mayman/projects/def-machado/mayman/agarle_bench/AgarLE-benchmark/PPO_multi_heads_full_action.py \
  --reward="reward_gym","min_max" \
  --lr=1e-5,3e-5,1e-4,3e-4 \
  --epochs=10,15,20 \
  --max-grad-norm=0.5,0.7,0.9
```

#### Step 2: Fine-Tuning with Best Parameters

After determining the best parameters from Step 1, refine the search with the following command:

```bash
python generate_bash.py PPO.sh /home/mayman/projects/def-machado/mayman/agarle_bench/AgarLE-benchmark/PPO_multi_heads_full_action.py \
  --reward="min_max" \
  --lr=0.00003 \
  --epochs=10 \
  --max-grad-norm=0.5 \
  --entropy-coef=0.05,0.01,0.1,0.5 \
  --clip-eps=0.2,0.4,0.5 \
  --value-func-coef=0.5,0.7,0.9
```

## Notes

- Ensure that all required dependencies are installed before running experiments.
- The `generate_bash.py` script automates job submission to Compute Canada.
- PPO tuning is performed in two stages to optimize performance efficiently.

## Contact

For any issues or improvements, feel free to raise an issue or contribute to this repository.
