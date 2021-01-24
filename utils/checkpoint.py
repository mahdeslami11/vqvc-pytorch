import torch
import os, glob

from .path import create_dir, get_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):

	if optimizer is not None:
		if not(os.path.exists(checkpoint_path)):
			print("[WARNING] No checkpoint exists. Start from scratch.")
			global_step = 0	
		else:
			print("[WARNING] Already exists. Restart to train model.")
			last_model_path = sorted(glob.glob(get_path(checkpoint_path, '*.pth.tar')))[-1]
			state = torch.load(last_model_path)

			model.load_state_dict(state['model'])
			global_step = state['global_step']
			optimizer.load_state_dict(state['optimizer'])
			scheduler.load_state_dict(state['scheduler'])
	else:
		last_model_path = sorted(glob.glob(get_path(checkpoint_path, '*.pth.tar')))[-1]
		state = torch.load(last_model_path)
		model.load_state_dict(state['model'])
		global_step = 0
		print("[WARNING] Model: {} has been loaded.".format(last_model_path.split("/")[-1].replace(".pth.tar", "")))	

	return global_step

		
def save_checkpoint(checkpoint_path, global_step, model, optimizer, scheduler):

	create_dir("/".join(checkpoint_path.split("/")[:-1]))
	checkpoint_path = create_dir(checkpoint_path)

	cur_checkpoint_name = "model-{:03d}k.pth.tar".format(global_step//1000)

	state = {
		'global_step': global_step,
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler.state_dict()
	}

	torch.save(state, get_path(checkpoint_path, cur_checkpoint_name))

