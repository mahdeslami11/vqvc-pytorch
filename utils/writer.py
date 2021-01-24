from torch.utils.tensorboard import SummaryWriter
from .path import create_dir
from .figure import draw_melspectrogram

class Writer(SummaryWriter):

	def __init__(self, log_path):
		super(Writer, self).__init__(log_path)
		create_dir("/".join(log_path.split("/")[:-1]))
		create_dir(log_path)


	def add_scalars(self, mode, global_step, loss):
		# edit
		self.add_scalar("{}".format(mode), loss, global_step)

	def add_mel_figures(self, mode, global_step, mel, mel_hat, mel_code, mel_style):
		figure = draw_melspectrogram(mel, mel_hat, mel_code, mel_style)
		self.add_figure("{}_mel(top)_mel_hat(top_mid)_code(bottom_mid)_style(bottom)".format(mode), figure, global_step)

