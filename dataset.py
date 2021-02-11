from config import Arguments as args
from utils.path import get_path
from utils.dataset import get_label_dictionary, normalize

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

	def __init__(self, mem_mode, meta_dir, dataset_name, mel_stat_path, max_frame_length=120):

		self.__mel_file_paths = self.__get_mel_filename(meta_dir=meta_dir)
		self.__label_dictionary = get_label_dictionary(dataset_name) 
		self.max_frame_length = max_frame_length
		self.mel_mean, self.mel_std = np.load(mel_stat_path).astype(np.float)

		if args.mem_mode:
			self.__mels = list(map(lambda mel_file_path: torch.tensor(np.load(mel_file_path)), self.__mel_file_paths))

	def __len__(self):
		return len(self.__mel_file_paths)

	def __getitem__(self, index):
		mel = self.__mels[index] if args.mem_mode else np.load(self.__mel_file_paths[index])

		T_mel, _ = mel.shape

		while T_mel <= self.max_frame_length:
			mel = torch.cat((mel, mel), dim=0)
			T_mel, _ = mel.shape

		index = np.random.randint(T_mel - self.max_frame_length + 1)		
		normalized_mel = normalize(mel[index: index + self.max_frame_length], mean=self.mel_mean, std=self.mel_std)

		return normalized_mel, 0


	def __get_mel_filename(self, meta_dir):	
		with open(meta_dir, "r") as f:
			mel_file_paths = list(map(lambda filename : filename.rstrip(), f.readlines()))
		return mel_file_paths



