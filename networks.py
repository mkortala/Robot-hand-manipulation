import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):

        output = torch.cat([state, action], 1)
        output = F.relu(self.input_layer(output))
        output = F.relu(self.hidden_layer(output))
        output = F.relu(self.hidden_layer2(output))
        output = self.output_layer(output)
        return output


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        output = F.relu(self.input_layer(state))
        output = F.relu(self.hidden_layer(output))
        output = F.relu(self.hidden_layer2(output))
        output = torch.tanh(self.output_layer(output))
        torch.no_grad()
        return output
