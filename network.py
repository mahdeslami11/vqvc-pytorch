import torch
import torch.nn as nn
import torch.nn.functional as F
import module as mm

class Encoder(nn.Module):
	"""
	Encoder
		Args:
			mel: (N, Tx, C_mel) log-melspectrogram (variable length)
		Returns:
			y_: (N, Tx, C_hidden)
	"""

	def __init__(self, mel_channels, channels, z_dim, kernel_size=3):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(
			mm.Conv1d(mel_channels, channels, kernel_size=kernel_size, 
					padding='same', bias=False, ln=True, activation_fn=nn.ReLU),
			mm.Conv1dResBlock(channels, channels, kernel_size, 
						padding='same', activation_fn=nn.ReLU, ln=True),
			mm.Conv1dResBlock(channels, channels, kernel_size, 
						padding='same', activation_fn=nn.ReLU, ln=True),
			mm.Conv1dResBlock(channels, channels, kernel_size, 
						padding='same', activation_fn=nn.ReLU, ln=True),
			mm.Conv1dResBlock(channels, channels, kernel_size, 
						padding='same', activation_fn=nn.ReLU, ln=True),
			mm.Conv1dResBlock(channels, channels, kernel_size, 
						padding='same', activation_fn=nn.ReLU, ln=True),
			
			mm.Linear(channels, z_dim, bias=False)
		)

	def forward(self, mels):
		z = self.encoder(mels)
		return z


class VQEmbeddingEMA(nn.Module):
	"""
		VQEmbeddingEMA
			- vector quantization module
			- ref
				from VectorQuantizedCPC official repository
				(https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py)

		encode:
			args:
				x:	(N, T, z_dim)
			returns:
				quantized:	(N, T, z_dim)
				indices:	(N, T)
		forward:
			args:
				x:	(N, T, z_dim)
			returns:
				quantized: (N, T, z_dim)
				loss:	(N, 1)
				perplexity: (N, 1)
	"""


	def __init__(self, n_embeddings, embedding_dim, decay=0.999, epsilon=1e-5):
		super(VQEmbeddingEMA, self).__init__()
		self.decay = decay
		self.epsilon = epsilon

		init_bound = 1 / n_embeddings
		embedding = torch.Tensor(n_embeddings, embedding_dim)
		embedding.uniform_(-init_bound, init_bound)
		self.register_buffer("embedding", embedding)
		self.register_buffer("ema_count", torch.zeros(n_embeddings))
		self.register_buffer("ema_weight", self.embedding.clone())

	def encode(self, x):
		M, D = self.embedding.size()
		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
				torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

		indices = torch.argmin(distances.float(), dim=-1)
		quantized = F.embedding(indices, self.embedding)
		quantized = quantized.view_as(x)
		return quantized, indices.view(x.size(0), x.size(1))

	def forward(self, x):
		M, D = self.embedding.size()
		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

		indices = torch.argmin(distances.float(), dim=-1)
		encodings = F.one_hot(indices, M).float()
		quantized = F.embedding(indices, self.embedding)
		quantized = quantized.view_as(x)

		if self.training:
			self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

			n = torch.sum(self.ema_count)
			self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

			dw = torch.matmul(encodings.t(), x_flat)
			self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
			self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

		commitment_loss = F.mse_loss(x, quantized.detach())

		quantized = x + (quantized - x).detach()

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		return quantized, commitment_loss, perplexity

class Decoder(nn.Module):
	"""
		Decoder

		args:
			z_enc:	(N, T, z_dim)
			z_quan:	(N, T, z_dim)
		return:
			mel_reconstructed:	(N, T, C_mel)	
	"""

	def __init__(self, in_channels, channels, mel_channels, kernel_size=3, norm_epsilon=1e-4, speaker_emb_discount_factor=3):
		super(Decoder, self).__init__()

		self.norm_epsilon = norm_epsilon
		self.speaker_emb_discount_factor = speaker_emb_discount_factor

		self.conv_layer = mm.Conv1d(in_channels, channels, kernel_size=kernel_size, bias=False)
		self.res_blocks = nn.Sequential(
					mm.Conv1dResBlock(channels, channels, 
							  kernel_size, activation_fn=nn.ReLU, ln=True),
					mm.Conv1dResBlock(channels, channels, 
							  kernel_size, activation_fn=nn.ReLU, ln=True),
					mm.Conv1dResBlock(channels, channels, 
							  kernel_size, activation_fn=nn.ReLU, ln=True),
					mm.Conv1dResBlock(channels, channels, 
							  kernel_size, activation_fn=nn.ReLU, ln=True),
					mm.Conv1dResBlock(channels, channels, 
							  kernel_size, activation_fn=nn.ReLU, ln=True))

		self.project_to_mel_dim = mm.Linear(channels, mel_channels, bias=False)


	def forward(self, z_enc, z_quan):

		speaker_emb = z_enc - z_quan
		code = z_quan

		# expectation over timestep
		normed_code = code / (torch.norm(code, dim=2, keepdim=True) + self.norm_epsilon)
		speaker_emb = torch.mean(speaker_emb, dim=1, keepdim=True)
		speaker_emb = speaker_emb / (torch.norm(speaker_emb, dim=1, keepdim=True) + self.norm_epsilon)/self.speaker_emb_discount_factor

		# 1e-4: avoid ZeroDivdedException, 3: intensity of speaker_embedding
		out = self.conv_layer(normed_code + speaker_emb)
		out = self.res_blocks(out)
		mel_reconstructed = self.project_to_mel_dim(out)

		return mel_reconstructed

	def evaluate(self, z_enc, z_quan):
		speaker_emb = z_enc - z_quan
		code = z_quan

		normed_code = code / (torch.norm(code, dim=2, keepdim=True) + self.norm_epsilon)
		speaker_emb = torch.mean(speaker_emb, dim=1, keepdim=True)

		speaker_emb = speaker_emb / (torch.norm(speaker_emb, dim=1, keepdim=True) + self.norm_epsilon)/self.speaker_emb_discount_factor

		# recon mel_hat 
		out = self.conv_layer(normed_code + speaker_emb)
		out = self.res_blocks(out)
		mel_reconstructed = self.project_to_mel_dim(out)

		# only code
		out = self.conv_layer(normed_code)
		out = self.res_blocks(out)
		mel_code = self.project_to_mel_dim(out)

		# only style
		out = self.conv_layer(z_enc - z_quan)
		out = self.res_blocks(out)
		mel_style = self.project_to_mel_dim(out)

		return mel_reconstructed, mel_code, mel_style


	def convert(self, src_contents, z_ref_enc, ref_contents):
		
		ref_speaker_emb = z_ref_enc - ref_contents

		normed_src_code = src_contents / (torch.norm(src_contents, dim=2, keepdim=True) + self.norm_epsilon)
		ref_speaker_emb = torch.mean(ref_speaker_emb, dim=1, keepdim=True)

		ref_speaker_emb = ref_speaker_emb / (torch.norm(ref_speaker_emb, dim=1, keepdim=True) + self.norm_epsilon)/self.speaker_emb_discount_factor

		# converted mel_hat 
		out = self.conv_layer(normed_src_code + ref_speaker_emb)
		out = self.res_blocks(out)
		mel_converted = self.project_to_mel_dim(out)

		# only src-code
		out = self.conv_layer(normed_src_code)
		out = self.res_blocks(out)
		mel_src_code = self.project_to_mel_dim(out)

		# only ref-style
		out = self.conv_layer(ref_speaker_emb)
		out = self.res_blocks(out)
		mel_ref_style = self.project_to_mel_dim(out)

		return mel_converted, mel_src_code, mel_ref_style

	
