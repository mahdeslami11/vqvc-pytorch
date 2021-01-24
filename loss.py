import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CPCLoss(nn.Module):

	"""
	Contrastive Predictive Coding Loss (CPCLoss)
	args:
		z: (N, T_mel, z_dim)
		c: (N, T_mel, c_dim)
	returns:
		

	"""


	def __init__(self, n_prediction_steps, batch_size, z_dim, c_dim):

		super(CPCLoss, self).__init__()

		self.n_prediction_steps = n_prediction_steps
		self.predictors = nn.ModuleList([
			nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
		])

		self.hidden = torch.zeros(1, batch_size, z_dim)
		self.softmax  = nn.Softmax()
		self.lsoftmax = nn.LogSoftmax()

	def forward(self, z_quan, c):

		B, T_mel, z_dim = z_quan.size()
		_, _, c_dim = c.size()

		nce = 0 # average over timestep and batch
		t_samples = torch.randint(T_mel - self.n_prediction_steps, size=(1,)).long() # randomly pick time stamps
		encode_samples = torch.empty((self.n_prediction_steps, B, z_dim)).float() # e.g. size 12*8*51
		pred = torch.empty((self.n_prediction_steps, B, z_dim)).float()
		c_t = c[:, t_samples, :].view(B, c_dim)

		for i in np.arange(1, self.n_prediction_steps + 1):
			encode_samples[i-1] = z_quan[:, t_samples+i, :].view(B, z_dim) # z_tk e.g. size 8*512
			linear = self.predictors[i-1]
			pred[i-1] = linear(c_t)
			total = torch.mm(encode_samples[i-1].clone(), torch.transpose(pred[i-1].clone(), 0, 1))
			correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, B)))
			nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor

		nce /= -1.*B*self.n_prediction_steps
		accuracy = 1.*correct.item()/B

		return accuracy, nce

class CrossEntropyLoss(nn.Module):
	def __init__(self):
		super(CrossEntropyLoss, self).__init__()

	def forward(self, net, labels):
		output = F.log_softmax(net, dim=-1)
		loss = F.nll_loss(output, labels)

		y_pred = output.max(1, keepdim=True)[1]
		accuracy = 1.*y_pred.eq(labels.view_as(y_pred)).sum().item()/len(y_pred)
		
		return loss, accuracy
