import numpy as np

def modify_action(logits, start, end):
    cont_action, dis_action = logits[:-1], logits[-1]
    range_array = np.linspace(start, end, 4)
    insert_index = np.searchsorted(range_array, dis_action)
    dis_action = insert_index - 1 
    dis_action = np.clip(dis_action, 0, 2)
    return ([(cont_action), dis_action])

def modify_hybrid_action(action):
    if action.ndim <= 1:
        cont_action, dis_action = action[:-1], action[-1]
    else:
        cont_action, dis_action = action[:, :-1][0], action[:, -1][0]
    
    dis_action = int(dis_action)
    return ([(cont_action), dis_action])