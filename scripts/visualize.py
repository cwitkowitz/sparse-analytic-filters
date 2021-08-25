# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from model import OnsetsFrames2LHVQT

# Regular imports
import torch
import os


def visualize(model, save_dir, i=None):
    """
    Visualize the filterbank.

    Parameters
    ----------
    model : OnsetsFramesLHVQT
      Model with a filterbank to visualize
    save_dir : string
      Directory under which to save filter images
    i : int
      Current iteration for directory organization
    """

    if i is not None:
        # Add an additional folder for the checkpoint
        save_dir = os.path.join(save_dir, f'checkpoint-{i}')

    # Visualize the filterbank
    model.frontend.fb.visualize(save_dir,
                                fix_scale=True,
                                include_axis=True,
                                n_fft=2**18,
                                decibels=True,
                                include_negative=True,
                                sort_by_centroid=True)


if __name__ == '__main__':
    model_path = 'path/to/model'
    save_dir = 'path/to/images'

    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}'
                          if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path, map_location=device)
    model.change_device(gpu_id)

    visualize(model, save_dir)
