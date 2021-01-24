import torch
import torch.nn as nn

class Linear(nn.Linear):
	"""
	Linear
	Args:
		x: (N, T, C_in)
	Returns:
		y: (N, T, C_out)
	"""

	def __init__(self, in_features, out_features, bias=True, activation_fn=None, ln=None, drop_rate=0.):
		super(Linear, self).__init__(in_features, out_features, bias=bias)

		self.activation_fn = activation_fn(inplace=True) if activation_fn is not None else None
		self.layer_norm = nn.LayerNorm(out_features) if ln is not None else None
		self.activation_fn = activation_fn(inplace=True) if activation_fn is not None else None
		self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None

	def forward(self, x):
		y = super(Linear, self).forward(x) 		
		y = self.layer_norm(y) if self.layer_norm is not None else y
		y = self.activation_fn(y) if self.activation_fn is not None else y
		y = self.drop_out(y) if self.drop_out is not None else y
		return y


class Conv1d(nn.Conv1d):
	"""
		Convolution 1d
		Args:
			x: (N, T, C_in)
		Returns:
			y: (N, T, C_out)
	"""

	def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                stride=1, padding='same', dilation=1, groups=1, bias=True, ln=False):

		if padding == 'same':
			padding = kernel_size // 2 * dilation
			self.even_kernel = not bool(kernel_size % 2)

		super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)

		self.activation_fn = activation_fn(inplace=True) if activation_fn is not None else None
		self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None
		self.layer_norm = nn.LayerNorm(out_channels) if ln else None

	def forward(self, x):
		y = x.transpose(1, 2)
		y = super(Conv1d, self).forward(y)
		y = y.transpose(1, 2)
		y = self.layer_norm(y) if self.layer_norm is not None else y
		y = self.activation_fn(y) if self.activation_fn is not None else y
		y = self.drop_out(y) if self.drop_out is not None else y
		y = y[:, :-1, :] if self.even_kernel else y
		return y

class Conv1dResBlock(Conv1d):
	"""
		Convolution 1d with Residual connection

		Args:
			x: (N, T, C_in)
		Returns:
			y: (N, T, C_out)
	"""
	def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                stride=1, padding='same', dilation=1, groups=1, bias=True, ln=False):

		super(Conv1dResBlock, self).__init__(in_channels, out_channels, kernel_size, activation_fn,
                                            drop_rate, stride, padding, dilation, groups=groups, bias=bias,
					    ln=ln)

	def forward(self, x):
		y = x
		y = y + super(Conv1dResBlock, self).forward(y)
		return y



