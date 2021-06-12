# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import MAESTRO_V3, MAPS

from amt_tools.train import validate
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
import torch

model_path = 'path/to/model'

gpu_id = 0
device = torch.device(f'cuda:{gpu_id}'
                      if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)

profile = model.profile
data_proc = model.lhvqt
hop_length = data_proc.hop_length
sample_rate = data_proc.sample_rate

# Initialize the estimation pipeline
validation_estimator = ComboEstimator([NoteTranscriber(profile=profile),
                                       PitchListWrapper(profile=profile)])

# Initialize the evaluation pipeline
evaluators = [LossWrapper(),
              MultipitchEvaluator(),
              NoteEvaluator(key=tools.KEY_NOTE_ON),
              NoteEvaluator(offset_ratio=0.2, key=tools.KEY_NOTE_OFF)]
validation_evaluator = ComboEvaluator(evaluators)

# Define expected path for calculated features and ground-truth
features_gt_cache = os.path.join('..', 'generated', 'data')

##################################################
# MAESTRO                                        #
##################################################

# Create a dataset corresponding to the MAESTRO testing partition
mstro_test = MAESTRO_V3(splits=['test'],
                        hop_length=hop_length,
                        sample_rate=sample_rate,
                        data_proc=data_proc,
                        profile=profile,
                        store_data=False,
                        save_loc=features_gt_cache)

# Get the average results for the MAESTRO testing partition
results = validate(model, mstro_test, evaluator=validation_evaluator, estimator=validation_estimator)

# Print the average results
print('MAESTRO Results')
print(results)

# Reset the evaluator
validation_evaluator.reset_results()

##################################################
# MAPS                                           #
##################################################

# Create a dataset corresponding to the MAPS testing partition
maps_test = MAPS(splits=['ENSTDkAm', 'ENSTDkCl'],
                 hop_length=hop_length,
                 sample_rate=sample_rate,
                 data_proc=data_proc,
                 profile=profile,
                 reset_data=False,
                 store_data=False,
                 save_loc=features_gt_cache)

# Get the average results for the MAPS testing partition
results = validate(model, maps_test, evaluator=validation_evaluator, estimator=validation_estimator)

# Print the average results
print('MAPS Results')
print(results)
