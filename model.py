from config import Arguments as args

import torch
import torch.nn as nn

from network import Encoder, VQEmbeddingEMA, Decoder
import module as mm

class VQVC(nn.Module):
	"""
		VQVC

		Args:
			mels:	(N, T, C)			

		Returns:
			encode:
				z_enc:		(N, T, z_dim)
				z_quan:		(N, T, z_dim)
				c:	 	(N, T, c_dim)
				indices: 	(N,
			forward:
				z_enc:		(N, T, z_dim)
				z_quan:		(N, T, z_dim)
				c: 		(N, T, c_dim)
				loss:		(1, )
				perplexity	(1, )
	"""

	def __init__(self):
		super(VQVC, self).__init__()
		self.name = 'VQVC'

		self.encoder = Encoder(mel_channels=args.n_mels, channels=args.encoder_hidden, z_dim=args.z_dim, kernel_size=args.encoder_kernel_size)   
		self.codebook = VQEmbeddingEMA(args.n_embeddings, args.z_dim)
		self.decoder = Decoder(in_channels=args.z_dim, channels=args.decoder_hidden, mel_channels=args.n_mels, kernel_size=args.decoder_kernel_size)



	def forward(self, mels):
		z_enc = self.encoder(mels)
		z_quan, commitment_loss, perplexity = self.codebook(z_enc)
		mels_hat = self.decoder(z_enc, z_quan)

		return mels_hat, commitment_loss, perplexity

	def evaluate(self, mels):
		z_enc = self.encoder(mels)
		z_quan, commitment_loss, perplexity = self.codebook(z_enc)
		mels_hat, mels_code, mels_style = self.decoder.evaluate(z_enc, z_quan)

		return mels_hat, mels_code, mels_style, commitment_loss, perplexity


	def convert(self, src_mel, ref_mel):
		z_src_enc = self.encoder(src_mel)
		src_contents, _, _ = self.codebook(z_src_enc)

		z_ref_enc = self.encoder(ref_mel)
		ref_contents, _, _ = self.codebook(z_ref_enc)
		
		mel_converted, mel_src_code, mel_ref_style = self.decoder.convert(src_contents, z_ref_enc, ref_contents)
		return mel_converted, mel_src_code, mel_ref_style
