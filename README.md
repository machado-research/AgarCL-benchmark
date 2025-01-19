# AgarLE Benchmarking

### Installation of the environment

#### Step 1: Add the submodules
```
git submodule init
git submodule update
```

#### Step 2: Create virtual environment

You can use conda or virtual environment. 

If you want to use a virtual environment: 
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Step 3: Install the environment as written in [AgarLE](https://github.com/machado-research/AgarLE.git)


### Running the benchmarking

For SAC or PPO run the following command:
```
python main.py --exp "the path of PPO or SAC" --idxs "The indices of number of runs"
```
For DQN, 
```
python train_rainbow.py 
```
