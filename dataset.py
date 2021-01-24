from config import Arguments as args
from utils.path import get_path
from utils.dataset import get_label_dictionary

import numpy as np

import torch
import glob
from torch.utils.data import dataset as dset

class SpeechDataset(dset.Dataset):
	"""
		class: Dataset

		returns:
			text: (Tx, Dim)
			mel:  (Ty, Dim)

		explanation:
			dataloader for training model
			mem_mode (in config.py): mags cannot be loaded onto the memory due to the large # of features
	"""

	def __init__(self, mem_mode, meta_dir, sample_frame, dataset_name):
		self.__mel_file_paths = self.__get_mel_filename(meta_dir=meta_dir)
		self.__label_dictionary = get_label_dictionary(dataset_name) 
		self.sample_frame = sample_frame

		if args.mem_mode:
			self.__mels = list(map(lambda mel_file_path: torch.tensor(np.load(mel_file_path)), self.__mel_file_paths))

	def __len__(self):
		return len(self.__mel_file_paths)

	def __getitem__(self, index):
		mel = self.__mels[index] if args.mem_mode else torch.tensor(np.load(self.__mel_file_paths[index]))
		T_mel, _ = mel.shape
		index = np.random.randint(T_mel - self.sample_frame + 1)

		# many-to-one mapping
		#labels = self.__get_label_index(self.__mel_file_paths[index], self.__label_dictionary)

		return mel[index: index+self.sample_frame], 0

	def __get_mel_filename(self, meta_dir):	
		with open(meta_dir, "r") as f:
			mel_file_paths = list(map(lambda filename : filename.rstrip(), f.readlines()))
		return mel_file_paths

	def __get_label_index(self, mel_file_path, label_dictionary):
		key = mel_file_path.split("/")[-1].split("_")[1]
		embedding_index = torch.tensor(label_dictionary[key]).long()
		return embedding_index







