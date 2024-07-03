# AgarLE Benchmarking

### Installation of the environment

#### Step 1: Add the submodules
```
git submodule init
git submodule update
```

#### Step 2: Create virtual environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Step 3: Install the environment as written in [AgarLE](https://github.com/machado-research/AgarLE.git)


### Running the benchmarking

For SAC run the following command:
```
python agar_sac.py
```

For PPO run the following command:
```
python agar_ppo.py
```