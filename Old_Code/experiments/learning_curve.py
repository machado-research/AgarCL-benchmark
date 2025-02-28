import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
import matplotlib.pyplot as plt

from PyExpPlotting.matplot import setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.intervals import bootstrap
from RlEvaluation.interpolation import compute_step_return

from experiment.ExperimentModel import ExperimentModel
from utils.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

COLORS = {
    'ppo': 'red',
    'sac': 'blue',
    'dqn': 'black',
}

# keep 1 in every SUBSAMPLE measurements
SUBSAMPLE = 100

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()
    plot_save_path = f'./plots'

    results = ResultCollection.fromExperiments(Model=ExperimentModel)
    hyper_cols = set[str]()
    for res in results:
        hyper_cols |= res.exp.get_hypers(0).keys()
    
    data_definition(
        hyper_cols=hyper_cols,
        seed_col='internal_seed',
        time_col='frame',
        environment_col='environment',
        algorithm_col='algorithm',

        # makes this data definition globally accessible
        # so we don't need to supply it to all API calls
        make_global=True,
    )

    df = results.combine(
        # converts path like "experiments/example/MountainCar"
        # into a new column "environment" with value "MountainCar"
        # None means to ignore a path part
        folder_columns=(None, 'environment'),

        # and creates a new column named "algorithm"
        # whose value is the name of an experiment file, minus extension.
        # For instance, ESARSA.json becomes ESARSA
        file_col='algorithm',
    )

    assert df is not None
    f, ax = plt.subplots()
    total_steps = results.get_any_exp().total_steps
    
    for agent in df.algorithm.unique():
        agent_df = df[df.algorithm == agent]
        agent_df = agent_df.groupby(['seed'])
        
        returns = np.zeros((len(agent_df), total_steps))
        for idx, (_, group) in enumerate(agent_df):
            step = group.steps.dropna().values
            reward = group.reward.dropna().values
            time = np.cumsum(step)
            _, data = compute_step_return(time, reward, total_steps)
            
            returns[idx] = data
        
        line = bootstrap(returns)
        
        lo = line[0]
        avg = line[1]
        hi = line[2]
        
        ax.plot(avg, label=agent, color=COLORS[agent], linewidth=0.8)
        ax.fill_between(range(line.shape[1]), lo, hi, color=COLORS[agent], alpha=0.25)

        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlabel('environment steps', fontsize=13)
        ax.set_ylabel('reward', fontsize=13)
        ax.set_title('Agar - default hyperparameters', fontsize=13)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        
    os.makedirs(f'{plot_save_path}/', exist_ok=True)
    plt.savefig(f'{plot_save_path}/square.png', bbox_inches="tight", dpi=300)
    print(f'Saved to {plot_save_path}/square.png')

    plt.close()

