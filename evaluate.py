from config import Arguments as args

from utils.path import create_dir, get_path

import torch

def evaluate(model, vocoder, eval_data_loader, criterion, global_step, writer=None, DEVICE=None):

	eval_path = create_dir(args.eval_path)
	model.eval()

	with torch.no_grad():
		eval_loss, eval_recon_loss, eval_perplexity, eval_commitment_loss = 0, 0, 0, 0

		for step, (mels, _) in enumerate(eval_data_loader):

			mels = mels.to(DEVICE)

			mels_hat, mels_code, mels_style, commitment_loss, perplexity = model.evaluate(mels.detach())

			commitment_loss = args.commitment_cost * commitment_loss
			recon_loss = criterion(mels, mels_hat)

			total_loss = commitment_loss + recon_loss

			eval_perplexity += perplexity.item()
			eval_recon_loss += recon_loss.item()
			eval_commitment_loss += commitment_loss.item()
			eval_loss += total_loss.item()

		mel =  mels[0].view(-1, args.n_mels).detach().cpu().numpy().T
		mel_hat = mels_hat[0].view(-1, args.n_mels).detach().cpu().numpy().T
		mel_code = mels_code[0].view(-1, args.n_mels).detach().cpu().numpy().T
		mel_style = mels_style[0].view(-1, args.n_mels).detach().cpu().numpy().T

		if args.log_tensorboard:
			writer.add_scalars(mode="eval_reconstruction_loss", global_step=global_step, loss=eval_recon_loss / len(eval_data_loader))
			writer.add_scalars(mode="eval_commitment_loss", global_step=global_step, loss=eval_commitment_loss / len(eval_data_loader))
			writer.add_scalars(mode="eval_perplexity", global_step=global_step, loss=eval_perplexity / len(eval_data_loader))
			writer.add_scalars(mode="eval_total_loss", global_step=global_step, loss=eval_loss / len(eval_data_loader))
			writer.add_mel_figures(mode="eval-mels_", global_step=global_step, mel=mel, mel_hat=mel_hat, mel_code=mel_code, mel_style=mel_style)



	
