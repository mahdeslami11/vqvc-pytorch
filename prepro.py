from config import Arguments as args
from utils.path import get_path, create_dir
from data import korean_emotional_speech_dataset

import codecs, sys
import numpy as np
import glob

def prepro_wavs():

	print("Start to preprocess {} wav signal...".format(args.dataset_name))

	dataset_path = args.dataset_path
	wav_dir = args.wav_dir

	create_dir(args.prepro_dir)
	prepro_path = create_dir(args.prepro_path)
	mel_path = create_dir(args.prepro_mel_dir)
	sampling_rate = args.sr
	sample_frame = args.sample_frame

	if "korean_emotional_speech" in args.dataset_name:
		korean_emotional_speech_dataset.preprocess(dataset_path, wav_dir, prepro_path, mel_path, sampling_rate, sample_frame, n_workers=args.n_workers)
	else:
		print("[ERROR] No Dataset named {}".format(args.dataset_name))


def write_meta():
	"""
		[TO DO] apply sampling based on audio duration when splitting train, eval, test dataset
	"""

	# split dataset into meta-train, meta-eval, meta-test with split ratio	
	print("[LOG] Start to split data with ratio:", args.data_split_ratio)

	assert np.sum(args.data_split_ratio) == 1., "sum of list data_split_ratio must be 1"

	meta_path = create_dir(args.prepro_meta_dir)

	file_list = glob.glob(get_path(args.prepro_mel_dir, "*.npy"))

	meta_train = codecs.open(args.prepro_meta_train, mode="w")
	meta_eval =  codecs.open(args.prepro_meta_eval, mode="w")

	train_num = int(len(file_list) * args.data_split_ratio[0])

	train_file = "\n".join(file_list[:train_num+1])
	eval_file = "\n".join(file_list[train_num+1:])

	meta_train.writelines(train_file)
	meta_eval.writelines(eval_file)

	print("[LOG] Done: split metadata")


if __name__ == "__main__":

	assert len(sys.argv) == 3, "[ERROR] # of args must be 1"

	_, is_wav, is_meta = sys.argv

	print("Audio signal: {}\t\tWrite metadata: {}".format(is_wav, is_meta))

	if is_wav in ["1", 1, "True"]:
		prepro_wavs()
	elif is_meta in ["1", 1, "True"]:
		write_meta()

