# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>n

# My imports
from amt_tools.datasets import MAESTRO_V3, MAPS

from amt_tools.train import train, validate
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

import lhvqt as l

from lhvqt_wrapper import LHVQT
from model import OnsetsFrames2LHVQT

from visualize import visualize
from sonify import sonify

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = 'hb+rnd+var'

ex = Experiment('Onsets & Frames 2 w/ LHVQT on MAESTRO')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 16000

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 625

    # Number of training iterations to conduct
    iterations = 200

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 10

    # Number of samples to gather for a batch
    batch_size = 8

    # The initial learning rate
    learning_rate = 6e-4

    # The id of the gpu to use, if available
    gpu_id = 1

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different feature extraction parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    expr_cache = os.path.join('..', 'generated', 'experiments')
    root_dir = os.path.join(expr_cache, EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def onsets_frames_run(sample_rate, hop_length, num_frames, iterations, checkpoints,
                      batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    tools.seed_everything(seed)

    # Initialize the default piano profile
    profile = tools.PianoProfile()

    # Processing parameters
    dim_in = 252
    model_complexity = 3

    # Initialize learnable filterbank data processing module
    data_proc = LHVQT(sample_rate=sample_rate,
                      hop_length=hop_length,
                      lhvqt=l.lhvqt_comb.LHVQT,
                      lvqt=l.lvqt_hilb.LVQT,
                      fmin=None,
                      harmonics=[1],
                      n_bins=dim_in,
                      bins_per_octave=36,
                      gamma=None,
                      max_p=1,
                      random=True,
                      update=True,
                      batch_norm=True,
                      var_drop=-10)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([NoteTranscriber(profile=profile),
                                           PitchListWrapper(profile=profile)])

    # Initialize the evaluation pipeline
    evaluators = [LossWrapper(),
                  MultipitchEvaluator(),
                  NoteEvaluator(key=tools.KEY_NOTE_ON),
                  NoteEvaluator(offset_ratio=0.2, key=tools.KEY_NOTE_OFF)]
    validation_evaluator = ComboEvaluator(evaluators, patterns=['loss', 'f1'])

    # Construct the MAESTRO splits
    train_split = ['train']
    val_split = ['validation']
    test_split = ['test']

    # Define expected path for raw datasets
    base_dir_mstro = None
    base_dir_maps = None

    # Define expected path for calculated features and ground-truth
    features_gt_cache = os.path.join('..', 'generated', 'data')

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    mstro_train = MAESTRO_V3(base_dir=base_dir_mstro,
                             splits=train_split,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             data_proc=data_proc,
                             profile=profile,
                             num_frames=num_frames,
                             reset_data=reset_data,
                             store_data=False,
                             save_data=True,
                             save_loc=features_gt_cache,
                             seed=seed)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=mstro_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)

    print('Loading validation partition...')

    # Create a dataset corresponding to the validation partition
    mstro_val = MAESTRO_V3(base_dir=base_dir_mstro,
                           splits=val_split,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           data_proc=data_proc,
                           profile=profile,
                           num_frames=num_frames,
                           reset_data=reset_data,
                           store_data=False,
                           save_data=True,
                           save_loc=features_gt_cache,
                           seed=seed)

    print('Loading testing partitions...')

    # Create a dataset corresponding to the MAESTRO testing partition
    mstro_test = MAESTRO_V3(base_dir=base_dir_mstro,
                            splits=test_split,
                            hop_length=hop_length,
                            sample_rate=sample_rate,
                            data_proc=data_proc,
                            profile=profile,
                            reset_data=reset_data,
                            store_data=False,
                            save_data=True,
                            save_loc=features_gt_cache)

    # Initialize the MAPS testing splits as the real piano data
    test_splits = ['ENSTDkAm', 'ENSTDkCl']

    # Create a dataset corresponding to the MAPS testing partition
    # Need to reset due to HTK Mel-Spectrogram spacing
    maps_test = MAPS(base_dir=base_dir_maps,
                     splits=test_splits,
                     hop_length=hop_length,
                     sample_rate=sample_rate,
                     data_proc=data_proc,
                     profile=profile,
                     reset_data=True,
                     store_data=False,
                     save_data=True,
                     save_loc=features_gt_cache)

    print('Initializing model...')

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames2LHVQT(dim_in, profile, data_proc.get_num_channels(), data_proc, model_complexity, True, gpu_id)
    onsetsframes.change_device()
    onsetsframes.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adam(onsetsframes.parameters(), learning_rate)

    # Define the visualization function with the root directory
    def vis_fnc(model, i):
        visualization_dir = os.path.join(root_dir, 'visualization')
        visualize(model, visualization_dir, i)

    # Visualize the filterbank before conducting any training
    vis_fnc(onsetsframes, 0)

    # Sonify the filterbank before conducting any training
    sonification_dir = os.path.join(root_dir, 'sonification', 'init')
    sonify(onsetsframes, sonification_dir)

    print('Training classifier...')

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    # Train the model
    onsetsframes = train(model=onsetsframes,
                         train_loader=train_loader,
                         optimizer=optimizer,
                         iterations=iterations,
                         checkpoints=checkpoints,
                         log_dir=model_dir,
                         val_set=mstro_val,
                         estimator=validation_estimator,
                         evaluator=validation_evaluator,
                         vis_fnc=vis_fnc)

    # Sonify the filterbank after training is complete
    sonification_dir = os.path.join(root_dir, 'sonification', 'final')
    sonify(onsetsframes, sonification_dir)

    print('Transcribing and evaluating test partition...')

    # Add save directories to the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated', 'MAESTRO'), ['notes', 'pitch'])

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results', 'MAESTRO'))
    validation_evaluator.set_patterns(None)

    # Get the average results for the MAESTRO testing partition
    results = validate(onsetsframes, mstro_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('MAESTRO Results', results, 0)

    # Reset the evaluator
    validation_evaluator.reset_results()

    # Update save directories for the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated', 'MAPS'), ['notes', 'pitch'])

    # Update save directory for the evaluators
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results', 'MAPS'))

    # Get the average results for the MAPS testing partition
    results = validate(onsetsframes, maps_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('MAPS Results', results, 0)
