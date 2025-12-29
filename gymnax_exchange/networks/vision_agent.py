import jax
import jax.numpy as jnp
from flax import linen as nn


class VisionAgent(nn.Module):
    """A Vision Agent that processes visual inputs and outputs actions.

    Attributes:
        num_actions: Number of possible actions the agent can take.
        hidden_size: Size of the hidden layers in the network.
    """
    num_actions: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        """Forward pass of the Vision Agent.

        Args:
            x: Input image tensor of shape (batch_size, height, width, channels).

        Returns:
            logits: Output logits for each action of shape (batch_size, num_actions).
        """
        # Convolutional layers to process the image input
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), activation=nn.relu)(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), activation=nn.relu)(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), activation=nn.relu)(x)

        # Flatten the output from convolutional layers
        x = x.reshape((x.shape[0], -1))

        # Fully connected layers
        x = nn.Dense(features=self.hidden_size, activation=nn.relu)(x)
        x = nn.Dense(features=self.hidden_size, activation=nn.relu)(x)

        # Output layer for action logits
        logits = nn.Dense(features=self.num_actions)(x)

        return logits