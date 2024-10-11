import itertools
import yaml
import sys
sys.path.append('./')
sys.path.append('../')


def get_configurations(params):
    # get all parameter configurations for individual runs
    list_params = [key for key in params.keys() if type(params[key]) is list]
    param_values = [params[key] for key in list_params]
    hyper_param_settings = list(itertools.product(*param_values))
    return list_params, hyper_param_settings


def sweeper():
    # Hypers settings
    # Env settings
    grid_size = [32, 64, 128, 256]

    # PPO settings
    gamma = [0.99]
    gae_lambda = [0.95]

    total_steps = [100000]
    rollout_length = [2048, 4096]
    epochs = [4, 10]
    num_mini_batch = [32, 64, 128, 256]

    clip_eps = [0.1, 0.2, 0.3]
    vf_coef = [0.3, 0.5, 0.7]
    gradient_clipping = [True]
    max_grad_norm = [0.5, 2.0, 5.0, 10.0]
    entropy_coef = [0.0, 1e-6, 1e-4]

    lrs = [1e-5, 1e-4, 3e-4, 1e-3]
    d_hidden = [64, 128, 256]

    seeds = [0, 1, 2, 3, 4]
    exp_description = 'agar-100k'
    wand_project = 'agar-ppo'
    params = {
        'seeds': seeds,  # 0
        'grid_size': grid_size,  # 1

        'gamma': gamma,  # 2
        'gae_lambda': gae_lambda,  # 3

        'total_steps': total_steps,  # 4
        'rollout_length': rollout_length,  # 5
        'epochs': epochs,  # 6
        'num_mini_batch': num_mini_batch,  # 7

        'clip_eps': clip_eps,  # 8
        'vf_coef': vf_coef,  # 9
        'gradient_clipping': gradient_clipping,  # 10
        'max_grad_norm': max_grad_norm,  # 11
        'entropy_coef': entropy_coef,  # 12

        'lr': lrs,  # 13
        'd_hidden': d_hidden,  # 14
    }
    list_params, hyper_param_settings = get_configurations(params)

    # Jobs settings
#     tasks_per_job = 1
#     # this is a relative path to the folder where all generated jobs will be saved in this folder
#     folder_name = 'jobs'
#     # relative path to the script to run
#     script_to_run = 'src/experiments/run-ppo.py'
#     # all generated config files will have this name at the begining
#     pre_config_name = 'agar-100k-ppo'
#     # cc configs
#     hours_per_job = 3
#     cpus_per_job = 8
#     mem_per_cpu = 10000  # in MB
#     account_name = 'rrg-whitem'

#     # writing jobs to files
#     f = open(folder_name+'/tasks_0.txt', 'w')
#     n_jobs_per_file = 0
#     n_files = 1
#     for config in hyper_param_settings:
#         file_data = {'tag': exp_description,
#                      'wandb_project_name': wand_project,
#                      'exp_seed': config[0],
#                      'env_config': {
#                          'ticks_per_step':  4,
#                          'num_frames':      1,
#                          'arena_size':      500,
#                          'num_pellets':     500,
#                          'num_viruses':     20,
#                          'num_bots':        10,
#                          'pellet_regen':    True,
#                          'grid_size':       config[1],
#                          'observe_cells':   False,
#                          'observe_others':  True,
#                          'observe_viruses': True,
#                          'observe_pellets': True,
#                          'obs_type': "grid",  # Two options: screen, grid
#                          'allow_respawn': True,  # If False, the game will end when the player is eaten
#                          # Two options: "mass:reward=mass", "diff = reward=mass(t)-mass(t-1)"
#                          'reward_type': 1,
#                          # reward = [diff or mass] - c_death if player is eaten
#                          'c_death': -100,
#                      },
#                      'ppo_config': {
#                          'gamma': config[2],
#                          'gae_lambda': config[3],
#                          'total_steps': config[4],
#                          'rollout_steps': config[5],
#                          'epochs': config[6],
#                          'num_mini_batch': config[7],
#                          'clip_eps': config[8],
#                          'vf_coef': config[9],
#                          'gradient_clipping': config[10],
#                          'max_grad_norm': config[11],
#                          'entropy_coef': config[12],
#                          # network related
#                          'lr': config[13],
#                          'd_hidden': config[14],
#                      },
#                      }
#         file_name = pre_config_name
#         for i in range(len(config)):
#             file_name += '_'+str(config[i])
#         with open('configs/'+file_name+'.yaml', 'w') as file:
#             yaml.dump(file_data, file, sort_keys=False)
            
#         cmd = 'python '+script_to_run+' --config_file=configs/'+file_name+'.yaml \n'
#         if n_jobs_per_file == tasks_per_job:
#             f.close()
#             f = open(folder_name+'/tasks_'+str(n_files)+'.txt', 'w')
#             n_jobs_per_file = 0
#             n_files += 1
#         f.write(cmd)
#         n_jobs_per_file += 1
#     f.close()

#     # writin cc bash files
#     # run single job bash file
#     f = open(folder_name+'/run_single_job.sh', 'w')
#     f.write('#!/bin/bash \n')
#     f.write('#SBATCH --account='+account_name+' \n')
#     f.write('#SBATCH --time='+str(hours_per_job)+':00:00 \n')
#     f.write('#SBATCH --cpus-per-task='+str(cpus_per_job)+' \n')
#     f.write('#SBATCH --mem-per-cpu='+str(mem_per_cpu)+'\n')
#     # f.write('#SBATCH --output='+folder_name+'/out_'+str(i)+'.txt \n')
#     # f.write('#SBATCH --error='+folder_name+'/err_'+str(i)+'.txt \n')
#     f.write('cd ~/projects/def-whitem/esraa/trace_memory/ \n')
#     f.write('source rtu/bin/activate \n')
#     f.write('wandb offline')
#     f.write('parallel  < $1 \n')
#     f.close()
#     # run all jobs bash file
#     f = open(folder_name+'/jobs.sh', 'w')
#     f.write('#!/bin/bash \n')
#     f.write('#SBATCH --account='+account_name+' \n')
#     f.write('#SBATCH --time=01:00:00 \n')
#     f.write('for FILE in /home/esraa/projects/def-whitem/esraa/trace_memory/jobs/*.txt \n \
#     do\n \
#         sbatch ~/projects/def-whitem/esraa/trace_memory/jobs/run_single_job.sh $FILE; \n \
#     done')
#     f.close()
#     return


# cmds = sweeper()
