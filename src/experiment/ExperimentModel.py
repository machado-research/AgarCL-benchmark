import sys
import json
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.agent = d.get('agent') 
        self.total_steps = d.get('total_steps')
        self.eval_steps = d.get('eval_steps')
        self.env_name = d.get('env_name')
        self.use_jax = d.get('use_jax')
        
        self.name = self.getExperimentName()

def load(path=None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp