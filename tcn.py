import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
								 self.conv2, self.chomp2, self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.ReLU()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)

class TemporalBlock_Single(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock_Single, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs[0], n_outputs[0], kernel_size[0],
										   stride=stride[0], padding=padding[0], dilation=dilation[0]))
		self.chomp1 = Chomp1d(padding[0])
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_inputs[1], n_outputs[1], kernel_size[1],
										   stride=stride[1], padding=padding[1], dilation=dilation[1]))
		self.chomp2 = Chomp1d(padding[1])
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
								 self.conv2, self.chomp2, self.relu2, self.dropout2)

		self.downsample = nn.Conv1d(n_inputs[0], n_outputs[-1], 1) if n_inputs[0] != n_outputs[-1] else None
		self.relu = nn.ReLU()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)

class TemporalConvNet_Single(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=[2,2], dropout=0.2):
		super(TemporalConvNet_Single, self).__init__()
		layers = []
		num_levels = len(out_channels)
		for i in range(num_levels):		
			dilation_size = [2 ** i, 2**(i+1)]
			layers += [TemporalBlock_Single(in_channels[i], out_channels[i], kernel_size[i], stride=[1,1], dilation=dilation_size,
								 padding=[(kernel_size[i][0]-1) *dilation_size[0] , (kernel_size[i][1]-1)*dilation_size[1]], dropout=dropout)] #padding is designed to make length fixed


		self.network = nn.Sequential(*layers)

	def forward(self, x):
		output = self.network(x)
		#print(output.shape)
		return output
