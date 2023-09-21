#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from pathlib import Path
from thop import profile, clever_format
import numpy as np
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm.contrib import tqdm
from speechbrain import Stage
import time
import torchaudio

class SpeakerBrain(sb.core.Brain):
    """Class for decider training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a whisper encoder + decider module.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id
        
        self.modules.whisper.encoder_only = True
        with torch.no_grad():
            enc_out = self.modules.whisper(wavs,bos_tokens) 
        
        enc_out = torch.permute(enc_out, (1,0, 3, 2)).to(self.device)

        if hasattr(self.modules.resnet,'ponderators'):
            self.modules.resnet.ponderators = self.modules.resnet.ponderators.to(self.device)
        
        lens = lens.to(self.device)
        outputs = self.modules.resnet(enc_out,lens,self.device)
        
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss: binary cross entropy.
        """
        predictions, lens = predictions
        uttid = batch.id
        
        GT = batch.choice
        GT = torch.Tensor(list(map(int,GT)))

        GT = GT.unsqueeze(1).float().to(self.device) 
        
        resSig = torch.nn.functional.sigmoid(predictions)
        loss = torch.nn.functional.binary_cross_entropy(resSig,GT)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, resSig.round(), GT, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None,test=False):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prepare(hparams,tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
        
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_datasets = {}
    test_datasets = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )
    test_datasets = test_datasets.filtered_sorted(
        sort_key="duration"
    )

    datasets = [train_data, valid_data,test_datasets] #+ [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav","id")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav,id):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd","id","choiceSmall")
    @sb.utils.data_pipeline.provides(
        "wrd","id","choice", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd,id,choice):
        yield wrd
        yield id
        yield choice
        tokens_list = tokenizer.encode(wrd)
        # avoid bos and eos tokens.
        tokens_list = tokens_list[1:-1]
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens


    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig","choice","tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets



if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    speaker_brain.modules.whisper.output_all_hiddens=True
 
    pp=0
    for p in list(speaker_brain.modules.resnet.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print(pp," parameters in the resnet")
    
    speaker_brain.tokenizer = tokenizer
    
    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"] #"dataloader_options"],
    )
    speaker_brain.evaluate(
        test_datasets, test_loader_kwargs=hparams["dataloader_options"]
    )
    
