from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(
        self,
        input_channels,
        sequence_length,
        output_dim,
        hidden_dim=500,
        num_conv_layers=6,
        conv_channels=[512, 512, 512, 1024, 2048, 1024],
        kernel_sizes=[5, 5, 7, 7, 7, 9],
        stride=1,
        padding=2,
    ):
        super(DQN, self).__init__()

        layers = []
        in_channels = input_channels
        for i in range(num_conv_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=conv_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = conv_channels[i]

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the output size after convolutions
        conv_out_dim = sequence_length
        for kernel_size in kernel_sizes:
            conv_out_dim = (conv_out_dim - kernel_size + 2 * padding) // stride + 1

        self.fc1 = nn.Linear(conv_channels[-1] * conv_out_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc3 = nn.Linear(hidden_dim * 4, hidden_dim * 8)
        self.fc4 = nn.Linear(hidden_dim * 8, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)

        # Flatten the output to feed into fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x
