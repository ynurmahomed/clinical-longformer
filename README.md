clinical-longformer
===================

This is the source code for the Master's thesis project "Predicting hospital readmission with long clinical notes".

Description
===========

The project aims to evaluate the effect of using a Transformer-based model with a sparse attention pattern to predict 30-day hospital readmission on a cohort from the MIMIC-III dataset.

Available here is the source-code for data processing, model training and evaluation.

The code was tested using Python 3.8 in a Linux environment, and it is recomended to use a [virtual environment](https://virtualenv.pypa.io/en/latest/).

Setup
=====

1. Install the dependencies with:

```
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

If you plan on using GPUS use the `reqirements-hpc.txt` files instead.

Data Processing
===============

To execute data processing, run the following module:
    
    python -m src.clinical_longformer.data.processing

To get the help messagae use the `--help` argument:

```
$ python -m src.clinical_longformer.data.processing --help
usage: processing.py [-h] [--n-days {1-30}] [-v] [-vv] mimic_path {ds,all} {-1,512,1024,2048,4096} [out_path]

Data processing

positional arguments:
mimic_path            MIMIC-III dataset path
{ds,all}              set notes category (ds - Discharge Summary)
{-1,512,1024,2048,4096}
                        set note length, -1 means do not chunk text
out_path              set output path

optional arguments:
-h, --help            show this help message and exit
--n-days {1-30}       set number of days (only used if category is set to all)
-v, --verbose         set loglevel to INFO
-vv, --very-verbose   set loglevel to DEBUG
```

Training
========

Model training is done with [PyTorch Lightning](https://www.pytorchlightning.ai/) framework.

There are four excutable modules available in `src/clinical_longformer/model`: `dan.py`, `lstm.py`, `bert.py` and `longformer.py`.
These modules run the Pytorch Lightning Trainer, you can find available arguments by using the `--help` argument. 
More information is available in the [docs](https://pytorch-lightning.readthedocs.io/en/1.5.2/common/trainer.html#trainer-flags).

In the case of `longformer.py` we are able to specify the maximum token length usgin the `--max_length` argument.

In the `hpc-uct`, and `chpc` folders there are examples of how to run the models.


Tuning
======

Hyperparameter tuning is done using [Weights & Biases](https://docs.wandb.ai/guides/sweeps).
Look inside `hpc-uct`, and `chpc` for examples of how to run the sweeps.


Pre-training
============

Pre-training is done using the HuggingFace's [Transformers library](https://huggingface.co/docs/transformers/index) language-modeling example [script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).

The script has been cloned to this [repository](https://github.com/ynurmahomed/language-modeling), where job files for executing in `hpc-uct` are available.
