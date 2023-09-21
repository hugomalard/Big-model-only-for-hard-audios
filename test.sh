#!/bin/bash
#SBATCH --job-name=trainCV 
#SBATCH -A jvn@v100
#SBATCH --gres=gpu:1 
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=6:00:00 
#SBATCH --output=%x-%j.log

source $HOME/.profile
export SETUPTOOLS_USE_DISTUTILS=stdlib
conda activate hugoenv

#inference of whisper models
python BMOHA/train_whisper.py BMOHA/hparams/whisper/train_en_hf_whisperTiny.yaml
python BMOHA/train_whisper.py BMOHA/hparams/whisper/train_en_hf_whisperSmall.yaml

python BMOHA/generate_labels.py
