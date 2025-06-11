import torch
import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead


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


class DQNNetwork(nn.Module):
    def __init__(self, obs_channels, act_dim, obs_hidden_dim, act_hidden_dim = 3, reward_hidden_dim= 2,  RNN_hidden_dim=128):
        super(DQNNetwork, self).__init__()

        self.obs_embedder = CustomCNN(obs_channels, obs_hidden_dim)
        self.prev_action_embedder = make_embedder(act_dim, act_hidden_dim)
        self.reward_embedder = make_embedder(1, reward_hidden_dim)

        self.rnn = nn.GRU(input_size=obs_hidden_dim + act_hidden_dim + reward_hidden_dim, hidden_size=RNN_hidden_dim, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(RNN_hidden_dim, RNN_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(RNN_hidden_dim, act_dim)  # Output Q-values
        )

    def forward(self, o_t, a_t_prev, r_t, hidden=None):
        e_obs = self.obs_embedder(o_t)
        e_prev_action = self.prev_action_embedder(a_t_prev)
        e_reward = self.reward_embedder(r_t)

        rnn_input = torch.cat([e_obs, e_prev_action, e_reward], dim=-1).unsqueeze(1)
        #Add LayerNorm to the input
        rnn_input = nn.LayerNorm(rnn_input.shape[-1]).to(rnn_input.device)(rnn_input)
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        rnn_out = nn.LayerNorm(rnn_out.shape[-1]).to(rnn_out.device)(rnn_out)

        q_values = self.mlp(rnn_out.squeeze(1))

        q_values = DiscreteActionValueHead(q_values)
        # Ensure q_values is a 2D tensor with shape (batch_size, num_actions)
        return q_values, hidden
