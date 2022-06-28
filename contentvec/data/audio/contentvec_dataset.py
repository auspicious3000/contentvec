# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np
import pickle

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils_1 import params2sos
from fairseq.data.audio.audio_utils_1 import change_gender
from fairseq.data.audio.audio_utils_1 import change_gender_f0
from fairseq.pdb import set_trace

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )
        


import parselmouth
import warnings
#warnings.filterwarnings("error")
from scipy.signal import sosfilt
Qmin, Qmax = 2, 5


class ContentvecDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        crop: bool = False,
        single_target: bool = False,
        spk2info = None
    ):
        self.split = manifest_path.split('/')[-1][:-4]
        assert self.split in ['train', 'valid']
        with open(spk2info, "rb") as f:
            spk2info = pickle.load(f)
        self.spk2info = spk2info[self.split]
        self.rng = np.random.default_rng()
        self.Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))
        
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.crop = crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, crop={crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )
        
    def random_eq(self, wav, sr):
        z = self.rng.uniform(0, 1, size=(10,))
        Q = Qmin * (Qmax / Qmin)**z
        G = self.rng.uniform(-12, 12, size=(10,))
        sos = params2sos(G, self.Fc, Q, sr)
        wav = sosfilt(sos, wav)
        return wav
    
    def random_formant_f0(self, wav, sr, spk):
        #s = parselmouth.Sound(wav, sampling_frequency=sr)
        _, (lo, hi, _) = self.spk2info[spk]
        
        if lo==50:
            lo=75
        if spk=="1447":
            lo, hi = 60, 400
        
        ratio_fs = self.rng.uniform(1, 1.4)
        coin = (self.rng.random() > 0.5)
        ratio_fs = coin*ratio_fs + (1-coin)*(1/ratio_fs)
        
        ratio_ps = self.rng.uniform(1, 2)
        coin = (self.rng.random() > 0.5)
        ratio_ps = coin*ratio_ps + (1-coin)*(1/ratio_ps)
        
        ratio_pr = self.rng.uniform(1, 1.5)
        coin = (self.rng.random() > 0.5)
        ratio_pr = coin*ratio_pr + (1-coin)*(1/ratio_pr)
        
        ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)
        
        return ss
    
    def fixed_formant_f0(self, wav, sr, spk):
        #s = parselmouth.Sound(wav, sampling_frequency=sr)
        _, (lo, hi, _) = self.spk2info[spk]
        
        if lo==50:
            lo=75
            ratio_fs, f0_med, ratio_pr = 1.2, 300, 1.2
        else:
            ratio_fs, f0_med, ratio_pr = 0.8, 100, 0.8
            
        ss = change_gender_f0(wav, sr, lo, hi, ratio_fs, f0_med, ratio_pr)
        
        return ss    

    def get_audio(self, index):
        import soundfile as sf
    
        fileName = self.audio_names[index]
        fileLen = self.sizes[index]
        spk = fileName.split('/')[1]
        wav_path = os.path.join(self.audio_root, fileName)
        wav, cur_sample_rate = sf.read(wav_path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if self.crop:
            wav = wav[:fileLen]
        if self.split == 'train':
            # 1st version
            try:
                wav_1 = self.random_formant_f0(wav, cur_sample_rate, spk)
            except UserWarning:
                wav_1 = np.copy(wav)
                print(f"Praat warining - {fileName}")
            except RuntimeError:
                wav_1 = np.copy(wav)
                print(f"Praat Error - {fileName}")
            wav_1 = self.random_eq(wav_1, cur_sample_rate)
            wav_1 = torch.from_numpy(wav_1).float()
            wav_1 = self.postprocess(wav_1, cur_sample_rate)
            # 2nd version
            try:
                wav_2 = self.random_formant_f0(wav, cur_sample_rate, spk)
            except UserWarning:
                wav_2 = np.copy(wav)
                print(f"Praat warining - {fileName}")
            except RuntimeError:
                wav_2 = np.copy(wav)
                print(f"Praat Error - {fileName}")
            wav_2 = self.random_eq(wav_2, cur_sample_rate)
            wav_2 = torch.from_numpy(wav_2).float()
            wav_2 = self.postprocess(wav_2, cur_sample_rate)
        elif self.split == 'valid':
            wav_1 = torch.from_numpy(wav).float()
            wav_1 = self.postprocess(wav_1, cur_sample_rate)
            try:
                wav_2 = self.fixed_formant_f0(wav, cur_sample_rate, spk)
            except UserWarning:
                wav_2 = np.copy(wav)
                print(f"Praat warining - {fileName}")
            except RuntimeError:
                wav_2 = np.copy(wav)
                print(f"Praat Error - {fileName}")
            wav_2 = torch.from_numpy(wav_2).float()
            wav_2 = self.postprocess(wav_2, cur_sample_rate)
        else:
            raise ValueError('Invalid dataset mode!')
        assert len(wav_1) == len(wav_2), "Different audio lengths!"
        spk_emb, _ = self.spk2info[spk]
        spk_emb = torch.from_numpy(spk_emb).float()
        return wav_1, wav_2, spk_emb

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav_1, wav_2, spk_emb = self.get_audio(index)
        labels = self.get_labels(index)
        return {"id": index, "source_1": wav_1, "source_2": wav_2, "label_list": labels, "spk_emb": spk_emb}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav_1, wav_2, target_size):
        size = len(wav_1)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav_1[start:end], wav_2[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source_1"] is not None]
        if len(samples) == 0:
            return {}

        audios_1 = [s["source_1"] for s in samples]
        audios_2 = [s["source_2"] for s in samples]
        audio_sizes = [len(s) for s in audios_1]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios_1, collated_audios_2, padding_mask, audio_starts = self.collater_audio(
            audios_1, audios_2, audio_size
        )
        
        spk_embs = [s["spk_emb"] for s in samples]
        collated_embs = self.collater_speaker(spk_embs)
        
        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source_1": collated_audios_1, "source_2": collated_audios_2, "padding_mask_1": padding_mask, "spk_emb": collated_embs}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch
    
    def collater_speaker(self, spk_embs):
        collated_speakers = torch.stack(spk_embs)
        return collated_speakers

    def collater_audio(self, audios_1, audios_2, audio_size):
        collated_audios_1 = audios_1[0].new_zeros(len(audios_1), audio_size)
        collated_audios_2 = audios_2[0].new_zeros(len(audios_2), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios_1.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios_1]
        for i, (audio_1, audio_2) in enumerate(zip(audios_1, audios_2)):
            diff = len(audio_1) - audio_size
            if diff == 0:
                collated_audios_1[i] = audio_1
                collated_audios_2[i] = audio_2
            elif diff < 0:
                assert self.pad_audio
                collated_audios_1[i] = torch.cat(
                    [audio_1, audio_1.new_full((-diff,), 0.0)]
                )
                collated_audios_2[i] = torch.cat(
                    [audio_2, audio_2.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios_1[i], collated_audios_2[i], audio_starts[i] = self.crop_to_max_size(
                    audio_1, audio_2, audio_size
                )
        return collated_audios_1, collated_audios_2, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                targets, lengths, ntokens = self.collater_seq_label(
                    targets, pad
                )
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        
        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav