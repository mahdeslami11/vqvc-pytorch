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

	def __init__(self, speaker_emb_reduction=3):
		super(VQVC, self).__init__()
		self.name = 'VQVC'

		self.speaker_emb_reduction = args.speaker_emb_reduction

		self.encoder = Encoder(mel_channels=args.n_mels, z_dim=args.z_dim)   
		self.codebook = VQEmbeddingEMA(args.n_embeddings, args.z_dim)
		self.decoder = Decoder(in_channels=args.z_dim, mel_channels=args.n_mels)

	def average_through_time(self, x, dim):
		x = torch.mean(x, dim=dim, keepdim=True)
		return x

	def forward(self, mels):

		# encoder
		z_enc = self.encoder(mels)

		# quantization
		z_quan, commitment_loss, perplexity = self.codebook(z_enc)

		# speaker emb
		speaker_emb_ = z_enc - z_quan
		speaker_emb = self.average_through_time(speaker_emb_, dim=1)

		# decoder
		mels_hat = self.decoder(z_quan, speaker_emb)

		return mels_hat, commitment_loss, perplexity

	def evaluate(self, mels):
		# encoder
		z_enc = self.encoder(mels)

		# contents emb
		z_quan, commitment_loss, perplexity = self.codebook(z_enc)

		# speaker emb
		speaker_emb_ = z_enc - z_quan
		speaker_emb =  self.average_through_time(speaker_emb_, dim=1)

		# decoder	
		mels_hat, mels_code, mels_style = self.decoder.evaluate(z_quan, speaker_emb, speaker_emb_)

		return mels_hat, mels_code, mels_style, commitment_loss, perplexity


	def convert(self, src_mel, ref_mel):
		# source z_enc
		z_src_enc = self.encoder(src_mel)

		# source contents
		src_contents, _, _ = self.codebook(z_src_enc)

		# source style emb
		src_style_emb_ = z_src_enc - src_contents

		# ref z_enc
		ref_enc = self.encoder(ref_mel)
	
		# ref contents
		ref_contents, _, _ = self.codebook(ref_enc)	

		# ref speaker emb
		ref_speaker_emb_ = ref_enc - ref_contents
		ref_speaker_emb = self.average_through_time(ref_speaker_emb_, dim=1)

		# decoder to generate mel
		mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style = self.decoder.convert(src_contents, src_style_emb_, ref_contents, ref_speaker_emb, ref_speaker_emb_)

		return mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style

