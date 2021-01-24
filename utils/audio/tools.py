"""
	from NVIDIA's preprocessing

	reference)
		https://github.com/NVIDIA/tacotron2
"""

import torch
import numpy as np

from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.signal as sps

import librosa
import os

import utils.audio.stft as stft
from utils.audio.audio_preprocessing import griffin_lim
from config import Arguments as args


_stft = stft.TacotronSTFT(
    args.filter_length, args.hop_length, args.win_length,
    args.n_mels, args.sr, args.mel_fmin, args.mel_fmax)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def get_mel(filename):
    audio, sampling_rate = load_wav_to_torch(filename)

    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} SR doesn't match target SR {}".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / args.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec, energy  = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0).detach().cpu().numpy().T
    energy = torch.squeeze(energy, 0).detach().cpu().numpy()

    return melspec, energy


def get_mel_from_wav(audio):
    sampling_rate = args.sr
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / args.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0).detach().cpu().numpy().T
    energy = torch.squeeze(energy, 0).detach().cpu().numpy()


    return melspec, energy



def inv_mel_spec(mel, out_filename, griffin_iters=60):
    mel = torch.stack([mel])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), _stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, args.sr, audio)
