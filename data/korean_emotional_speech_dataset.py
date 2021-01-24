from utils.path import *
from utils.audio.tools import get_mel

from tqdm import tqdm
import numpy as np
import glob, os, sys
from multiprocessing import Pool

from scipy.io.wavfile import write
import librosa, ffmpeg


def job(wav_filename):

	original_wav_filename, prepro_wav_dir, sampling_rate = wav_filename
	filename = original_wav_filename.split("/")[-1]
	new_wav_filename = get_path(prepro_wav_dir, filename)

	if not os.path.exists(new_wav_filename):
		try:
			out, err = (ffmpeg
					.input(original_wav_filename)
					.output(new_wav_filename, acodec='pcm_s16le', ac=1, ar=sampling_rate)
					.overwrite_output()
					.run(capture_stdout=True, capture_stderr=True))

		except ffmpeg.Error as err:
			print(err.stderr, file=sys.stderr)
			raise


def preprocess(data_path, prepro_wav_dir, prepro_path, mel_path, sampling_rate, sample_frame, n_workers=10):
	p = Pool(n_workers)

	index = 1

	prepro_wav_dir = create_dir(prepro_wav_dir)
	wav_paths=[[filename, prepro_wav_dir, sampling_rate] for filename in list(glob.glob(get_path(data_path, "**", "wav", "*.wav")))]

	print("\t[LOG] converting wav format...")
	with tqdm(total=len(wav_paths)) as pbar:
		for _ in tqdm(p.imap_unordered(job, wav_paths)):
			pbar.update()	

	print("\t[LOG] saving mel-spectrogram...")
	with tqdm(total=len(wav_paths)) as pbar:
		for wav_filename in tqdm(glob.glob(get_path(prepro_wav_dir, "*.wav"))):
			mel_filename = wav_filename.split("/")[-1].replace("wav", "npy")
			mel_savepath = get_path(mel_path, mel_filename)
			mel_spectrogram, _ = get_mel(wav_filename)

			T_mel, _ = mel_spectrogram.shape

			if T_mel > sample_frame:
				np.save(mel_savepath, mel_spectrogram)

	print("Done!")

