from utils.path import *
from utils.audio.tools import get_mel

from tqdm import tqdm
import numpy as np
import glob, os, sys
from multiprocessing import Pool

from scipy.io.wavfile import write
import librosa, ffmpeg
from sklearn.preprocessing import StandardScaler

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


def preprocess(data_path, prepro_wav_dir, prepro_path, mel_path, sampling_rate, n_workers=10, filter_length=1024, hop_length=256, trim_silence=True, top_db=60):
	p = Pool(n_workers)

	mel_scaler = StandardScaler(copy=False)

	prepro_wav_dir = create_dir(prepro_wav_dir)
	wav_paths=[[filename, prepro_wav_dir, sampling_rate] for filename in list(glob.glob(get_path(data_path, "wav48", "**", "*.wav")))]

	print("\t[LOG] converting wav format...")
	with tqdm(total=len(wav_paths)) as pbar:
		for _ in tqdm(p.imap_unordered(job, wav_paths)):
			pbar.update()	

	print("\t[LOG] saving mel-spectrogram...")
	with tqdm(total=len(wav_paths)) as pbar:
		for wav_filename in tqdm(glob.glob(get_path(prepro_wav_dir, "*.wav"))):
			mel_filename = wav_filename.split("/")[-1].replace("wav", "npy")
			mel_savepath = get_path(mel_path, mel_filename)
			mel_spectrogram, _ = get_mel(wav_filename, trim_silence=trim_silence, frame_length=filter_length, hop_length=hop_length, top_db=top_db)

			mel_scaler.partial_fit(mel_spectrogram)
			np.save(mel_savepath, mel_spectrogram)

	np.save(get_path(prepro_path, "mel_stats.npy"), np.array([mel_scaler.mean_, mel_scaler.scale_]))

	print("Done!")



def split_unseen_speakers(prepro_mel_dir):

	print("[LOG] 6 UNSEEN speakers:  \n\t p226(Male, English, Surrey) \n\t p256(Male, English, Birmingham) \
					 \n\t p266(Female, Irish, Athlone) \n\t p297(Female, American, Newyork) \
					 \n\t p323 (Female, SouthAfrican, Pretoria)\n\t p376(Male, Indian)")

	unseen_speaker_list = ["p226", "p256", "p266", "p297", "p323", "p376"]

	seen_speaker_files, unseen_speaker_files = [], []

	preprocessed_file_list = glob.glob(get_path(prepro_mel_dir, "*.npy"))
	
	for preprocessed_mel_file in preprocessed_file_list:
		speaker = preprocessed_mel_file.split("/")[-1].split("_")[0]
		if speaker in unseen_speaker_list:
			unseen_speaker_files.append(preprocessed_mel_file)
		else:
			seen_speaker_files.append(preprocessed_mel_file)	
	
	return seen_speaker_files, unseen_speaker_files




