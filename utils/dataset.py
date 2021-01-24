from .path import *
from .audio.tools import get_mel

import os


def get_label_dictionary(dataset_name):
	if "korean_emotional_speech" in dataset_name:
		return {'ang': 0, 'fea': 1, 'neu': 2, 'sur': 3, 'dis': 4, 'hap':5, 'sad': 6}
	else:
		return None

def get_src_and_ref_mels(src_path, ref_path):
	src_mel, ref_mel = None, None

	if os.path.isfile(src_path) and os.path.isfile(ref_path):
		src_mel, _  = get_mel(src_path)
		ref_mel, _ = get_mel(ref_path)
	else:
		print("[ERROR] No paths exist! Check your filename.: \n\t src_path: {}   ref_path: {}".format(src_path, ref_path))

	return src_mel, ref_mel

