#!/usr/bin/env python3
import os
from convolutionWhisperSpace import ResNet1DPonderate
import sys
import torch
import logging
import speechbrain as sb
from speechbrain import Stage
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import undo_padding
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm.contrib import tqdm
from thop import profile, clever_format
import numpy as np
import torchaudio

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the transcription using the model chosen by the decider."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id
        macsPre = 0
        macsTot = 0 
        self.modules.decider.ponderators = self.modules.decider.ponderators.to(self.device)

       
        self.modules.whisperMedium.encoder_only = True
        self.modules.whisperTiny.encoder_only = True
       
        #Compute the forward pass on the encoder of Whisper Small and the decider
        with torch.no_grad(): 
            macsEncoder, params = profile(self.modules.whisperMedium, inputs=(wavs, bos_tokens), verbose =False) 
            macsPre += macsEncoder 
                
            enc_out = self.modules.whisperMedium(wavs, bos_tokens)
            enc_out2 = torch.permute(enc_out, (1,0, 3, 2)).to(self.device)
            macsDecider, params = profile(self.modules.decider, inputs=(enc_out2,wav_lens,self.device), verbose =False)
            macsPre += macsDecider
            
            decision = torch.sigmoid(self.modules.decider(enc_out2,wav_lens,self.device))
            with open('recDecider2.txt', "a") as fileSave:
                fileSave.write(str(batch.id[0])+" "+str(decision.item()) + "\n")
                
            enc_out2 = None

            self.macsPreDec.append(macsPre)

         #Compute the forward pass on the decoder of Whisper Small if it has been chosen
        if decision.item() > 0.5:
            self.modules.whisperMedium.encoder_only = False
            macsPost, params = profile(self.hparams.test_beam_searcher_medium, inputs = (enc_out[-1], wav_lens),verbose =False)
            hyps, _ = self.hparams.test_beam_searcher_medium(enc_out[-1], wav_lens)
            macsTot = macsPre + macsPost

        #Or compute the forward on the encoder and decoder of Whisper Tiny otherwise
        else:
            del enc_out
            macsTinyEnc,params =  profile(self.modules.whisperTiny, inputs=(wavs, bos_tokens), verbose =False)    
            macsPre += macsTinyEnc
                
            enc_out = self.modules.whisperTiny(wavs, bos_tokens)
            self.modules.whisperTiny.encoder_only = False
            macsPost, params = profile(self.hparams.test_beam_searcher_tiny, inputs = (enc_out, wav_lens),verbose =False)
            hyps, _ = self.hparams.test_beam_searcher_tiny(enc_out, wav_lens)
            macsTot =  macsPost + macsPre
                
             
        self.macsTot.append(macsTot)
        log_probs = enc_out

        return (log_probs, hyps, wav_lens),decision > 0.5
    
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """

        out,modelUsed = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu(),modelUsed

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        nbBigModel = 0
        nbIte = 0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss,modelUsed = self.evaluate_batch(batch, stage=Stage.TEST)
                nbBigModel += modelUsed
                nbIte += 1 
                #if nbIte == 10:
                 #   break
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break
                
                
            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            )
        self.step = 0

        nbSmall = nbIte-nbBigModel

        print('Smaller model used ',nbIte-nbBigModel,'times \n')
        print('Bigger model used ',nbBigModel,'times')

        return avg_test_loss

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        log_probs, hyps, wav_lens, = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss = torch.Tensor([0])
        tokens, tokens_lens = batch.tokens

        # Decode token terms to words
        predicted_words = self.tokenizer.batch_decode(
            hyps, skip_special_tokens=True
        )

        # Convert indices to words
        target_words = undo_padding(tokens, tokens_lens)
        target_words = self.tokenizer.batch_decode(
            target_words, skip_special_tokens=True
        )

        if hasattr(self.hparams, "normalized_transcripts"):
            predicted_words = [
                self.tokenizer._normalize(text).split(" ")
                for text in predicted_words
            ]

            target_words = [
                self.tokenizer._normalize(text).split(" ")
                for text in target_words
            ]
        else:
            predicted_words = [text.split(" ") for text in predicted_words]

            target_words = [text.split(" ") for text in target_words]
        self.wer_metric.append(ids, predicted_words, target_words)
        self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.cer_metric = self.hparams.cer_computer()
        self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        stage_stats["CER"] = self.cer_metric.summarize("error_rate")
        stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        self.hparams.train_logger.log_stats(
            stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
            test_stats=stage_stats,
        )
        with open(self.hparams.wer_file, "w") as w:
            self.wer_metric.write_stats(w)


def dataio_prepare(hparams, tokenizer):
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

    
    # test is separate
    test_datasets = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams['test_csv'], replacements={"data_root": data_folder}
    )
    test_datasets = test_datasets.filtered_sorted(
        sort_key="duration"
    )
    
    datasets = [train_data, valid_data, test_datasets]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
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
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
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


    # Defining tokenizer and loading it
    tokenizer = hparams["whisperTiny"].tokenizer
    tokenizer.set_prefix_tokens(hparams["language"], "transcribe", False)

    # we need to prepare the tokens for searchers
    #hparams["valid_greedy_searcher"].set_decoder_input_tokens(
     #   tokenizer.prefix_tokens
    #)
    #hparams["valid_greedy_searcher"].set_language_token(
    #    tokenizer.prefix_tokens[1]
    #)

    hparams["test_beam_searcher_tiny"].set_decoder_input_tokens(
        tokenizer.prefix_tokens
    )
    hparams["test_beam_searcher_tiny"].set_language_token(tokenizer.prefix_tokens[1])

    hparams["test_beam_searcher_medium"].set_decoder_input_tokens(
        tokenizer.prefix_tokens
    )
    hparams["test_beam_searcher_medium"].set_language_token(tokenizer.prefix_tokens[1])

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )
    
    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    CKPT = torch.load(hparams['decider_path'])
    asr_brain.modules.decider.load_state_dict(CKPT,strict=False)
    asr_brain.modules.decider.ponderators = asr_brain.modules.decider.ponderators.to(asr_brain.device)
    asr_brain.macsPreDec = []
    asr_brain.macsTot = []
    print("MACs will be computed on evaluation")
    asr_brain.modules.whisperMedium.output_all_hiddens=True 

    # Testing
        
       
    asr_brain.hparams.wer_file = os.path.join(
        hparams["output_folder"], "wer_decider.txt"
    )
    asr_brain.evaluate(
        test_datasets, test_loader_kwargs=hparams["test_loader_kwargs"]
    )
    macsPre = clever_format([np.mean(asr_brain.macsPreDec)], "%.3f") #HERE
    print(f" Mean number of macs pre-decision: {macsPre}")
    macsTot = clever_format([np.mean(asr_brain.macsTot)], "%.3f")
    print(f" Mean number of macs total: {macsTot}")
