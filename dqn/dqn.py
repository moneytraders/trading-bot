import torch
from torch import nn
import torch.nn.functional as F

# # class DQN(nn.Module):

# #     def __init__(
# #             self,
# #             input_channels,
# #             sequence_length,
# #             output_dim,
# #             hidden_dim=1064,  # Increased hidden dimension
# #             num_conv_layers=6,  # More convolutional layers
# #             conv_channels=[64, 256, 256, 1064, 1064,
# #                            20128],  # More channels in each conv layer
# #             kernel_sizes=[5, 5, 5, 5, 5, 5],  # Larger kernels
# #             stride=1,
# #             padding=2):  # Increased padding to maintain feature map size
# #         super(DQN, self).__init__()

# #         # Create a list of convolutional layers
# #         layers = []
# #         in_channels = input_channels
# #         for i in range(num_conv_layers):
# #             layers.append(
# #                 nn.Conv1d(in_channels=in_channels,
# #                           out_channels=conv_channels[i],
# #                           kernel_size=kernel_sizes[i],
# #                           stride=stride,
# #                           padding=padding))
# #             layers.append(nn.ReLU())  # Apply ReLU after each convolution
# #             in_channels = conv_channels[i]

# #         self.conv_layers = nn.Sequential(*layers)

# #         # Calculate the output size after convolutions
# #         conv_out_dim = sequence_length
# #         for kernel_size in kernel_sizes:
# #             conv_out_dim = (conv_out_dim - kernel_size +
# #                             2 * padding) // stride + 1

# #         # Define fully connected layers with much larger hidden size
# #         self.fc1 = nn.Linear(conv_channels[-1] * conv_out_dim, hidden_dim)
# #         self.fc2 = nn.Linear(hidden_dim,
# #                              hidden_dim * 2)  # Even larger second FC layer
# #         self.fc3 = nn.Linear(hidden_dim * 2,
# #                              hidden_dim * 4)  # Third larger FC layer
# #         self.fc4 = nn.Linear(hidden_dim * 4, output_dim)  # Final output layer

# #     def forward(self, x):
# #         # Pass through convolutional layers
# #         x = self.conv_layers(x)

# #         # Flatten the output to feed into fully connected layers
# #         x = x.view(x.size(0), -1)

# #         # Pass through fully connected layers
# #         x = self.fc1(x)
# #         x = F.relu(x)
# #         x = self.fc2(x)
# #         x = F.relu(x)
# #         x = self.fc3(x)
# #         x = F.relu(x)
# #         x = self.fc4(x)

# #         return x

import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(
            self,
            input_channels,
            sequence_length,
            output_dim,
            hidden_dim=500,  # Increased hidden dimension
            num_conv_layers=5,  # More convolutional layers
            conv_channels=[128, 256, 512, 900, 1600],  # Even more channels
            kernel_sizes=[5, 5, 7, 7, 7],  # Larger kernels
            stride=1,
            padding=2):  # Increased padding to maintain feature map size
        super(DQN, self).__init__()

        # Create a list of convolutional layers
        layers = []
        in_channels = input_channels
        for i in range(num_conv_layers):
            layers.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=conv_channels[i],
                          kernel_size=kernel_sizes[i],
                          stride=stride,
                          padding=padding))
            layers.append(nn.ReLU())  # Apply ReLU after each convolution
            in_channels = conv_channels[i]

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the output size after convolutions
        conv_out_dim = sequence_length
        for kernel_size in kernel_sizes:
            conv_out_dim = (conv_out_dim - kernel_size +
                            2 * padding) // stride + 1

        # Define fully connected layers with much larger hidden size
        # Increase the size of the fully connected layers to reflect the increased channels
        self.fc1 = nn.Linear(conv_channels[-1] * conv_out_dim,
                             hidden_dim * 2)  # Larger FC layer
        self.fc2 = nn.Linear(hidden_dim * 2,
                             hidden_dim * 4)  # Even larger second FC layer
        self.fc3 = nn.Linear(hidden_dim * 4,
                             hidden_dim * 8)  # Third larger FC layer
        self.fc4 = nn.Linear(hidden_dim * 8, output_dim)  # Final output layer

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Flatten the output to feed into fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x
