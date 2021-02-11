import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt

def draw_melspectrogram(mel, mel_hat, mel_code, mel_style):
	fig, axis = plt.subplots(4, 1, figsize=(20,30))

	axis[0].set_title("Ground-truth mel")
	axis[0].imshow(mel, origin="lower", aspect="auto")

	axis[1].set_title("Reconstructed mel")
	axis[1].imshow(mel_hat, origin="lower", aspect="auto")

	axis[2].set_title("Contents mel")
	axis[2].imshow(mel_code, origin="lower", aspect="auto")

	axis[3].set_title("Style mel (normed_z - normed_z_quan")
	axis[3].imshow(mel_style, origin="lower", aspect="auto")

	return fig

def draw_converted_melspectrogram(src_mel, ref_mel, mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style):
	fig, axis = plt.subplots(5, 1, figsize=(20,30))

	axis[0].set_title("Source mel")
	axis[0].imshow(src_mel, origin="lower", aspect="auto")

	axis[1].set_title("Reference mel")
	axis[1].imshow(ref_mel, origin="lower", aspect="auto")

	axis[2].set_title("Converted mel")	
	axis[2].imshow(mel_converted, origin="lower", aspect="auto")

	axis[3].set_title("Source contents mel")
	axis[3].imshow(mel_src_code, origin="lower", aspect="auto")

	axis[4].set_title("Source style mel")
	axis[4].imshow(mel_src_style, origin="lower", aspect="auto")

	axis[5].set_title("Reference contents mel")
	axis[5].imshow(mel_ref_code, origin="lower", aspect="auto")

	axis[6].set_title("Reference style  mel")
	axis[6].imshow(mel_ref_style, origin="lower", aspect="auto")

	return fig
