import torch
import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.nn.recurrent import Recurrent
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

# Optional embedder for inputs like rewards, previous actions
def make_embedder(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU()
    )

# Custom CNN to use as observation embedder
class CustomCNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, activation=nn.LeakyReLU(), bias=0.1):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
                nn.LayerNorm([32, 31, 31]),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.LayerNorm([64, 14, 14]),
                nn.Conv2d(64, 32, 3, stride=1),
                nn.LayerNorm([32, 12, 12]),
            ]
        )
        self.output = nn.Linear(32 * 12 * 12, n_output_channels) 

        self.apply(self.init_chainer_default)
        self.apply(self.constant_bias_initializer(bias=bias))

    def constant_bias_initializer(self, bias=0.1):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, bias)
        return init

    def init_chainer_default(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))


class MultiInputEmbeddingModule(nn.Module):
    def __init__(self, obs_channels, act_dim, obs_hidden_dim_a, obs_hidden_dim_b, act_hidden_dim, reward_hidden_dim):
        super().__init__()
        self.obs_embedder_a = CustomCNN(obs_channels, obs_hidden_dim_a)
        self.obs_embedder_b = CustomCNN(obs_channels, obs_hidden_dim_b)
        self.prev_action_embedder = make_embedder(act_dim, act_hidden_dim)
        self.reward_embedder = make_embedder(1, reward_hidden_dim)
        self.act_dim = act_dim

    def forward(self, obs, action, reward):
        action = action[0]
        reward = reward[0]

        # One-hot encode
        action = action.to(obs.device).long()
        batch_size = action.size(0)
        action_onehot = torch.zeros(batch_size, self.act_dim, device=obs.device)
        action_onehot.scatter_(1, action.view(-1, 1), 1)

        e_obs_a = self.obs_embedder_a(obs)
        e_obs_b = self.obs_embedder_b(obs)
        e_action = self.prev_action_embedder(action_onehot)
        device = obs.device  # or use: next(self.parameters()).device
        reward = reward.view(-1, 1).to(device)
        e_reward = self.reward_embedder(reward)

        return torch.cat([e_obs_a, e_obs_b, e_action, e_reward], dim=-1)




class GRURecurrent(Recurrent, nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.ln_in = nn.LayerNorm(input_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, x, h, action, reward):
        was_packed = isinstance(x, PackedSequence)

        if was_packed:
            padded, lengths = pad_packed_sequence(x, batch_first=True)
            padded = self.ln_in(padded)
            x = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        else:
            x = self.ln_in(x)

        out, h = self.rnn(x, h)

        if was_packed:
            padded, lengths = pad_packed_sequence(out, batch_first=True)
            padded = self.ln_out(padded)
            out = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        else:
            out = self.ln_out(out)

        return out, h



class MLPActorHead(nn.Module):
    def __init__(self, hidden_size, act_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.dqn_head = DiscreteActionValueHead()
    def forward(self, x, action, reward):
        x = x.squeeze(1)
        q_values = self.mlp(x)
        return self.dqn_head(q_values)


# class DQNNetwork(nn.Module):
#     def __init__(self, obs_channels, act_dim, obs_hidden_dim, act_hidden_dim = 3, reward_hidden_dim= 2,  RNN_hidden_dim=128):
#         super(DQNNetwork, self).__init__()

#         self.obs_embedder = CustomCNN(obs_channels, obs_hidden_dim)
#         self.prev_action_embedder = make_embedder(act_dim, act_hidden_dim)
#         self.reward_embedder = make_embedder(1, reward_hidden_dim)
#         self.act_dim = act_dim
#         self.rnn = nn.GRU(input_size=obs_hidden_dim + act_hidden_dim + reward_hidden_dim, hidden_size=RNN_hidden_dim, batch_first=True)

#         self.mlp = nn.Sequential(
#             nn.Linear(RNN_hidden_dim, RNN_hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(RNN_hidden_dim, act_dim)  # Output Q-values
#         )
#         self.dqn_head = DiscreteActionValueHead()

#     def forward(self, o_t, a_t_prev, r_t, hidden=None):
    
        
#         a_t_prev = a_t_prev[0]
#         r_t = r_t[0]
        
#         # Convert a_t_prev to a tensor on the same device as o_t
#         a_t_prev_tensor = a_t_prev.to(o_t.device).long()  # shape: [batch]

#         # One-hot encode
#         batch_size = a_t_prev_tensor.shape[0]
#         a_t_prev_onehot = torch.zeros(batch_size, self.act_dim, device=o_t.device)
#         a_t_prev_onehot.scatter_(1, a_t_prev_tensor.view(-1, 1), 1)
 
#         a_t_prev = a_t_prev_onehot
        
#         e_obs = self.obs_embedder(o_t)
#         e_prev_action = self.prev_action_embedder(a_t_prev)
#         e_reward = self.reward_embedder(r_t.view(-1,1).to(o_t.device))


#         rnn_input = torch.cat([e_obs, e_prev_action, e_reward], dim=-1).unsqueeze(1)
#         #Add LayerNorm to the input
#         rnn_input = nn.LayerNorm(rnn_input.shape[-1]).to(rnn_input.device)(rnn_input)
#         rnn_out, hidden = self.rnn(rnn_input, hidden)
#         rnn_out = nn.LayerNorm(rnn_out.shape[-1]).to(rnn_out.device)(rnn_out)

#         q_values = self.mlp(rnn_out.squeeze(1))

#         q_values = self.dqn_head(q_values)
#         # Ensure q_values is a 2D tensor with shape (batch_size, num_actions)
#         return q_values, rnn_out
