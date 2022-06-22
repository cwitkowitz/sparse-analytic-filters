## Sparse Analytic Filters (Piano Transcription)
Code for the paper [Learning Sparse Analytic Filters for Piano Transcription](https://arxiv.org/abs/2108.10382).
The repository contains scripts to run and analyze a filterbank learning experiment on a piano transcription task.
It is a thin layer on top of two more general repositories:
 - [lhvqt](https://github.com/cwitkowitz/lhvqt) - filterbank learning module
 - [amt-tools](https://github.com/cwitkowitz/amt-tools) - music transcription framework

## Installation
Clone the repository and install the requirements.
```
git clone https://github.com/cwitkowitz/sparse-analytic-filters
pip install -r sparse-analytic-filters/requirements.txt
```

This will install the two aforementioned repositories along with dependencies.

## Usage
#### Experiment
The full training-testing pipeline can be run from the command line as follows:
```
python scripts/experiment.py
```

Parameters for the experiment are defined towards the top of the script.
These include hyperparameters such as sampling rate or learning rate, as well as filterbank parameters.

#### Transcription Model
The transcription model is defined in ```scripts/model.py```.
This script is where modifications can be made, e.g. Bernoulli dropout.
It is also where KL-divergence scaling or annealing can be set.

#### Analysis of Saved Models
##### Visualization
The filterbank of a saved model checkpoint can be visualized manually as follows:
```
python scripts/visualize.py
```

The GPU to use, the model path, and the directory under which to save images are all defined at the bottom of the script.

##### Sonification
The filterbank of a saved model checkpoint can be sonified manually as follows:
```
python scripts/sonify.py
```

The GPU to use, the model path, and the directory under which to save audio are all defined at the bottom of the script.

##### Evaluation
A saved model checkpoint can be evaluated manually as follows:
```
python scripts/evaluate.py
```

The GPU to use and the model path are defined at the top of the script.

## Generated Files
The experiment root directory ```<root_dir>``` is one parameter defined at the top of the experiment script.
Execution of ```scripts/experiment.py``` will generate the following under ```<root_dir>```:
 - ```n/```

    Folder (beginning at ```n = 1```) containing [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment files:
 
     - ```config.json``` - parameter values for the experiment
     - ```cout.txt``` - contains any text printed to console
     - ```metrics.json``` - evaluation results for the experiment
     - ```run.json``` system and experiment information

    An additional folder (```n += 1```) with experiment files is created for each run where the name of the [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment is the same. 

 - ```models/```

    Folder containing saved model and optimizer states at checkpoints, as well as the events file that tensorboard reads.

 - ```estimated/```

    Folder containing frame-level and note-level predictions for all tracks in the test set.
    Predictions are organized within ```.txt``` files according to the [MIREX I/O format](https://www.music-ir.org/mirex/wiki/2020:Multiple_Fundamental_Frequency_Estimation_%26_Tracking) for transcription.

 - ```results/```

    Folder containing individual evaluation results for each track within the test set.

 - ```sonification/```

    Folder containing sonified filters at initialization and after training completion.

 - ```_sources/```

    Folder containing copies of the script at the time(s) of execution.

 - ```visualization/```

    Folder containing visualized filters at initialization and every training checkpoint thereafter.

Additionally, ground-truth will be saved under the path specified by ```features_gt_cache```, unless ```save_data=False```.

## Analysis
During training, losses and various validation metrics can be analyzed in real-time by running:
```
tensorboard --logdir=<root_dir>/models --port=<port>
```
Here we assume the current directory within the command-line interface contains ```<root_dir>```.
 ```<port>``` is an integer corresponding to an available port (```port = 6006``` if unspecified).

After running the command, navigate to <http://localhost:port> to view any reported training or validation observations within the tensorboard interface.

## Cite
##### SMC 2022 Paper
```
@inproceedings{cwitkowitz2022learning,
  title     = {Learning Sparse Analytic Filters for Piano Transcription},
  author    = {Frank Cwitkowitz and Mojtaba Heydari and Zhiyao Duan},
  year      = 2022,
  booktitle = {Proceedings of Sound and Music Computing Conference (SMC)}
}
```
