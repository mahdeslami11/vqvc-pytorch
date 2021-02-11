from config import Arguments as args

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.conversion_visible_devices

import sys
import torch
from tqdm import tqdm
import numpy as np

from model import VQVC

from utils.dataset import get_src_and_ref_mels, normalize, de_normalize
from utils.vocoder import get_vocgan, vocgan_infer
from utils.path import get_path, create_dir
from utils.checkpoint import load_checkpoint
from utils.figure import draw_converted_melspectrogram

from config import Arguments as args


def convert(model, vocoder, mel_stat, conversion_wav_paths, DEVICE=None):


	for idx, (src_wav_path, ref_wav_path) in tqdm(enumerate(conversion_wav_paths), total=len(conversion_wav_paths), unit='B', ncols=70, leave=False):

		mel_mean, mel_std = mel_stat

		src_mel, ref_mel = get_src_and_ref_mels(src_wav_path, ref_wav_path, trim_silence=args.trim_silence, frame_length=args.filter_length, hop_length=args.hop_length, top_db=args.top_db)
		src_mel, ref_mel = normalize(src_mel, mel_mean, mel_std), normalize(ref_mel, mel_mean, mel_std)

		src_mel = torch.from_numpy(src_mel).float().unsqueeze(0).to(DEVICE)
		ref_mel = torch.from_numpy(ref_mel).float().unsqueeze(0).to(DEVICE)

		mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style = model.convert(src_mel, ref_mel)

		src_wav_name = src_wav_path.split("/")[-1]
		ref_wav_name = ref_wav_path.split("/")[-1]

		src_wav_path = get_path(args.converted_sample_dir, "{}_src_{}".format(idx, src_wav_name))
		ref_wav_path = get_path(args.converted_sample_dir, "{}_ref_{}".format(idx, ref_wav_name))
		converted_wav_path = get_path(args.converted_sample_dir, "{}_converted_{}_{}".format(idx, src_wav_name.replace(".wav", ""), ref_wav_name)) 		
		src_code_wav_path = get_path(args.converted_sample_dir, "{}_src_code_{}".format(idx, src_wav_name))
		src_style_wav_path = get_path(args.converted_sample_dir, "{}_src_style_{}".format(idx, src_wav_name))
		ref_code_wav_path = get_path(args.converted_sample_dir, "{}_ref_code_{}".format(idx, ref_wav_name))
		ref_style_wav_path = get_path(args.converted_sample_dir, "{}_ref_style_{}".format(idx, ref_wav_name))

		mel_mean, mel_std = torch.from_numpy(mel_mean).float().to(DEVICE), torch.from_numpy(mel_std).float().to(DEVICE)


		src_mel = de_normalize(src_mel, mel_mean, mel_std)
		ref_mel = de_normalize(ref_mel, mel_mean, mel_std)
		mel_converted = de_normalize(mel_converted, mel_mean, mel_std)
		mel_src_code = de_normalize(mel_src_code, mel_mean, mel_std)
		mel_src_style = de_normalize(mel_src_style, mel_mean, mel_std)
		mel_ref_code = de_normalize(mel_ref_code, mel_mean, mel_std)
		mel_ref_style = de_normalize(mel_ref_style, mel_mean, mel_std) 

		vocgan_infer(src_mel.transpose(1, 2), vocoder, path=src_wav_path)
		vocgan_infer(ref_mel.transpose(1, 2), vocoder, path = ref_wav_path)
		vocgan_infer(mel_converted.transpose(1, 2), vocoder, path = converted_wav_path)
		vocgan_infer(mel_src_code.transpose(1, 2), vocoder, path = src_code_wav_path)
		vocgan_infer(mel_src_style.transpose(1, 2), vocoder, path= src_style_wav_path)
		vocgan_infer(mel_ref_code.transpose(1, 2), vocoder, path= ref_code_wav_path)
		vocgan_infer(mel_ref_style.transpose(1, 2), vocoder, path = ref_style_wav_path)

		src_mel = src_mel.transpose(1, 2).squeeze().detach().cpu().numpy()
		ref_mel = ref_mel.transpose(1, 2).squeeze().detach().cpu().numpy()
		mel_converted = mel_converted.transpose(1, 2).squeeze().detach().cpu().numpy()
		mel_src_code = mel_src_code.transpose(1, 2).squeeze().detach().cpu().numpy()
		mel_src_style = mel_src_style.transpose(1, 2).squeeze().detach().cpu().numpy()
		mel_ref_code = mel_ref_code.transpose(1, 2).squeeze().detach().cpu().numpy()
		mel_ref_style = mel_ref_style.transpose(1, 2).squeeze().detach().cpu().numpy()

		fig = draw_converted_melspectrogram(src_mel, ref_mel, mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style)
		fig.savefig(get_path(args.converted_sample_dir, "contents_{}_style_{}.png".format(src_wav_name.replace(".wav", ""), ref_wav_name.replace(".wav", ""))))	


def main(DEVICE):

	# load model
	model = VQVC().to(DEVICE)
	vocoder = get_vocgan(ckpt_path=args.vocoder_pretrained_model_path).to(DEVICE)

	load_checkpoint(args.model_checkpoint_path, model)
	mel_stat = np.load(args.mel_stat_path)

	dataset_root = args.wav_dir

	src_paths = [get_path(dataset_root, "p226_354.wav"), get_path(dataset_root, "p225_335.wav")]
	ref_paths = [get_path(dataset_root, "p225_335.wav"), get_path(dataset_root, "p226_354.wav")]

	create_dir(args.converted_sample_dir)

	convert(model, vocoder, mel_stat, conversion_wav_paths=tuple(zip(src_paths, ref_paths)), DEVICE=DEVICE)


if __name__ == "__main__":

	print("[LOG] Start conversion...")

	DEVICE = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

	main(DEVICE)	
	print("[LOG] Finish..")
