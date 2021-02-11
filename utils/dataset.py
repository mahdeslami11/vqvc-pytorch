from .path import *
from .audio.tools import get_mel

import numpy as np
import os
import torch

def get_label_dictionary(dataset_name):
	if "korean_emotional_speech" in dataset_name:
		return {'ang': 0, 'fea': 1, 'neu': 2, 'sur': 3, 'dis': 4, 'hap':5, 'sad': 6}
	else:
		return None

def get_src_and_ref_mels(src_path, ref_path, trim_silence=True, frame_length=1024, hop_length=1024, top_db=10):
	src_mel, ref_mel = None, None

	if os.path.isfile(src_path) and os.path.isfile(ref_path):
		src_mel, _  = get_mel(src_path, trim_silence=trim_silence, frame_length=frame_length, hop_length=hop_length, top_db=top_db)
		ref_mel, _ = get_mel(ref_path, trim_silence=trim_silence, frame_length=frame_length, hop_length=hop_length, top_db = top_db)
	else:
		print("[ERROR] No paths exist! Check your filename.: \n\t src_path: {}   ref_path: {}".format(src_path, ref_path))

	return src_mel, ref_mel

def normalize(x, mean, std):
	zero_idxs = np.where(x==0.0)[0]
	z = (x - mean) / std
	z[zero_idxs] = 0.0
	return z

def de_normalize(z, mean, std):
	zero_idxs = torch.where(z == 0.0)[0]
	x = mean + std * z
	x[zero_idxs] = 0.0
	return x
