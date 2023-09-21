#!/bin/bash
#inference of whisper models
python BMOHA/train_whisper.py BMOHA/hparams/whisper/train_en_hf_whisperTiny.yaml
python BMOHA/train_whisper.py BMOHA/hparams/whisper/train_en_hf_whisperSmall.yaml

python BMOHA/generate_labels.py
