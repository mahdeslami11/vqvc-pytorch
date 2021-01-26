import sys
import torch
from tqdm import tqdm

from model import VQVC

from utils.dataset import get_src_and_ref_mels
from utils.vocoder import get_vocgan, vocgan_infer
from utils.path import get_path, create_dir
from utils.checkpoint import load_checkpoint

from config import Arguments as args


def convert(model, vocoder, conversion_wav_paths, DEVICE=None):

	for idx, (src_wav_path, ref_wav_path) in tqdm(enumerate(conversion_wav_paths), total=len(conversion_wav_paths), unit='B', ncols=70, leave=False):

		src_mel, ref_mel = get_src_and_ref_mels(src_wav_path, ref_wav_path)
		src_mel = torch.tensor(src_mel).unsqueeze(0).to(DEVICE)
		ref_mel = torch.tensor(ref_mel).unsqueeze(0).to(DEVICE)

		mel_converted, mel_src_code, mel_ref_style = model.convert(src_mel, ref_mel)

		src_wav_name = src_wav_path.split("/")[-1]
		ref_wav_name = ref_wav_path.split("/")[-1]

		src_wav_path = get_path(args.converted_sample_dir, "{}_src_{}".format(idx, src_wav_name))
		ref_wav_path = get_path(args.converted_sample_dir, "{}_ref_{}".format(idx, ref_wav_name))
		converted_wav_path = get_path(args.converted_sample_dir, "{}_converted_{}_{}".format(idx, src_wav_name.replace(".wav", ""), ref_wav_name)) 		
		code_wav_path = get_path(args.converted_sample_dir, "{}_code_{}".format(idx, src_wav_name))
		style_wav_path = get_path(args.converted_sample_dir, "{}_style_{}".format(idx, ref_wav_name))

		vocgan_infer(src_mel.transpose(1, 2), vocoder, path=src_wav_path)
		vocgan_infer(ref_mel.transpose(1, 2), vocoder, path = ref_wav_path)
		vocgan_infer(mel_converted.transpose(1, 2), vocoder, path = converted_wav_path)
		vocgan_infer(mel_src_code.transpose(1, 2), vocoder, path = code_wav_path)
		vocgan_infer(mel_ref_style.transpose(1, 2), vocoder, path = style_wav_path)

def main(DEVICE):

	# load model
	model = VQVC().to(DEVICE)
	vocoder = get_vocgan(ckpt_path=args.vocoder_pretrained_model_path).to(DEVICE)

	load_checkpoint(args.model_checkpoint_path, model)

	dataset_root = args.wav_dir

	src_paths = [get_path(dataset_root, "acriil_neu_00000479.wav")]
	ref_paths = [get_path(dataset_root, "acriil_sad_00002044.wav")]

	create_dir(args.converted_sample_dir)

	convert(model, vocoder, conversion_wav_paths=tuple(zip(src_paths, ref_paths)), DEVICE=DEVICE)


if __name__ == "__main__":

	print("[LOG] Start conversion...")

	gpu_ids = sys.argv[1]

	DEVICE = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

	main(DEVICE)	
	print("[LOG] Finish..")
