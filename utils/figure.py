import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt

def draw_melspectrogram(mel, mel_hat, mel_code, mel_style):
	fig, axis = plt.subplots(4, 1, figsize=(20,30))
	axis[0].imshow(mel, origin="lower", aspect="auto")
	axis[1].imshow(mel_hat, origin="lower", aspect="auto")
	axis[2].imshow(mel_code, origin="lower", aspect="auto")
	axis[3].imshow(mel_style, origin="lower", aspect="auto")

	return fig

