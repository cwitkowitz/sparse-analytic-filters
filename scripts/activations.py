# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from model import OnsetsFrames2LHVQT

import amt_tools.tools as tools

# Regular imports
import torch

model_path = 'path/to/model'
audio_path = 'path/to/audio'
save_dir = 'path/to/images'

gpu_id = 0
device = torch.device(f'cuda:{gpu_id}'
                      if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)

lvqt = model.lhvqt.lhvqt.get_modules()[0]

audio, _ = tools.load_normalize_audio(audio_path, lvqt.fs)

audio = torch.Tensor(audio).to(device)

lvqt.get_activations(save_dir, audio, include_axis=True, fix_scale=True)
