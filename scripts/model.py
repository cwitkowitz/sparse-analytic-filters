# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>n

# My imports
from amt_tools.models import OnsetsFrames2

import amt_tools.tools as tools

# Regular imports
import torch


def kld_scaling(iter):
    """
    Define a function with which to scale the KL-divergence loss term
    as a function of the current iteration.

    Parameters
    ----------
    iter : int
      Current training iteration

    Returns
    ----------
    scaling : float
      KL-divergence scaling factor for the current iteration
    """

    scaling = 0.01

    return scaling


class OnsetsFrames2LHVQT(OnsetsFrames2):
    """
    Implements the Onsets & Frames model (V2) with a learnable filterbank frontend.
    """

    def __init__(self, dim_in, profile, in_channels, lhvqt, model_complexity=2, detach_heads=True, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See OnsetsFrames2 class for others...

        lhvqt : LHVQT (Wrapper)
          Filterbank to use as frontend
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, detach_heads, device)

        # Create a pointer to the wrapper
        self.lhvqt = lhvqt

        # Append the filterbank learning module to the front of the model
        self.frontend.add_module('fb', self.lhvqt.lhvqt)
        self.frontend.add_module('rl', torch.nn.ReLU())

    def post_proc(self, batch):
        """
        Calculate KL-divergence for the 1D convolutional layer in the filterbank and append to the tracked loss.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multi pitch, onsets, and offsets output as well as loss
        """

        # Perform standard Onsets & Frames 2 steps
        output = super().post_proc(batch)

        # Obtain a pointer to the filterbank module
        fb_module = self.frontend.fb.get_modules()[0]

        # Check to see if loss is being tracked
        if tools.KEY_LOSS in output.keys() and fb_module.var_drop:
            # Extract all of the losses
            loss = output[tools.KEY_LOSS]

            # Calculate the KL-divergence term
            loss[tools.KEY_LOSS_KLD] = fb_module.time_conv.kld()

            # Compute the total loss and add it back to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] += kld_scaling(self.iter) * loss[tools.KEY_LOSS_KLD]
            output[tools.KEY_LOSS] = loss

        return output
