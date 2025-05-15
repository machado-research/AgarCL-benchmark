# Hyperparameter Tuning for DQN, PPO, and SAC

This repository provides comprehensive instructions to reproduce the results from our study on hyperparameter tuning for DQN, PPO, and SAC in hybrid-action environments. The algorithms were adapted to address the unique challenges posed by such environments:

- **PPO and SAC**: Modified using the PFRL framework to handle hybrid actions.
- **DQN**: Continuous action space discretized into 24 discrete actions, derived from 8 predefined directions combined with 3 discrete action types.

## Predefined Directions

The 8 predefined directions used for discretization are:

- **Up** \((0, 1)\)
- **Up-Right** \((1, 1)\)
- **Right** \((1, 0)\)
- **Down-Right** \((1, -1)\)
- **Down** \((0, -1)\)
- **Down-Left** \((-1, -1)\)
- **Left** \((-1, 0)\)
- **Up-Left** \((-1, 1)\)

These adaptations were critical for enabling the algorithms to operate effectively in the hybrid-action environment. For further details, refer to the accompanying paper.

---

## Setup Instructions

### 1. Environment Setup (Compute Canada)

If using Compute Canada, load the required modules with the following commands:

```bash
module load clang/17.0.6
module load cmake
module load python/3.11.5
module load scipy-stack
module load glfw
module load cuda/12.2
module load opencv/4.11.0
```

For other environments, refer to the [AgarCL installation guide](https://github.com/AgarCL/AgarCL).

### 2. Install Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Install AgarCL

Navigate to the AgarLE directory and install the package:

```bash
cd AgarCL_DIR
python setup.py install --user  # or
pip install -e .
```

> **Note:** These installations are one-time processes and do not need to be repeated for subsequent experiments.

---

## Running Hyperparameter Tuning

The hyperparameter tuning process is automated using scripts that facilitate job submission to cluster environments.

### DQN Hyperparameter Tuning

Use the following command to generate and submit the DQN tuning script:

```bash
python SubmitParallelJobs.py DQN.sh YOUR_DIR/DQN_full_action_set.py\
  --lr=1e-5,3e-5,1e-4,3e-4\
  --batch_accumulator="sum","mean"\
  --minibatch_size=32,64\
  --tau=1e-2,1e-3,5e-3\
  --seed=0,1,2
```

### SAC Hyperparameter Tuning

Run the following command for SAC tuning:

```bash
python generate_bash.py SAC.sh SAC_full_action_set.py\
  --reward="reward_gym","min_max"\
  --lr=3e-5,1e-4,1e-5\
  --soft-update-tau=0.01,0.005,0.001\
  --temperature-lr=1e-4,1e-5\
  --max-grad-norm=0.5,0.7,0.9\
  --seed=0,1,2
```

### PPO Hyperparameter Tuning

Run the following command for PPO tuning:

```bash
python generate_bash.py PPO.sh /home/mayman/projects/def-machado/mayman/agarle_bench/AgarLE-benchmark/PPO_multi_heads_full_action.py \
  --reward="reward_gym","min_max" \
  --lr=1e-5,3e-5,1e-4,3e-4\
  --epochs=10,15,20\
  --max-grad-norm=0.5,0.7,0.9\
  --entropy-coef=0.05,0.01,0.1,0.5\
  --clip-eps=0.2,0.4,0.5\
  --value-func-coef=0.5,0.7,0.9\
  --seed=0,1,2
```

### Post-Tuning Analysis

After running the tuning scripts, use the `get_best_hyper` notebook to identify the best hyperparameters based on the metric of averaging the last 100 steps.

---

## Notes

- Ensure all dependencies are installed before running experiments.
- The `generate_bash.py` script simplifies job submission to Compute Canada.
- PPO tuning involves a two-stage process for efficient optimization.

---

## Contact

For issues or suggestions, raise an issue or contribute to this repository.
