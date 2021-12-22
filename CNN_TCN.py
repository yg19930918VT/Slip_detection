import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn, vgg16_bn, vgg11_bn, inception_v3, alexnet, resnet18, resnet34, resnet50, wide_resnet50_2
import numpy as np
from tcn import TemporalConvNet_Single
import pdb

class BasicConv2d(nn.Module):
	def __init__(self,in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv2d = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
		self.relu = nn.ReLU()
	
	def forward(self,x):
		x = self.conv2d(x)
		x = self.relu(x)
		
		return x
	
class CNN_block(nn.Module): #two sensor and action concatenate model
	def __init__(self):
		super(CNN_block,self).__init__()
		self.conv2dlayer1_ID5 = BasicConv2d(3,16,kernel_size = (3,3), stride=(1,1), padding=(1,1))
		self.conv2dlayer2_ID5 = BasicConv2d(16,32,kernel_size = (2,2), stride=(1,1), padding=(0,0))
		self.conv2dlayer1_ID4 = BasicConv2d(3,16,kernel_size = (3,3), stride=(1,1), padding=(1,1))
		self.conv2dlayer2_ID4 = BasicConv2d(16,32,kernel_size = (2,2), stride=(1,1), padding=(0,0))
		self.fclayer_cat = BasicConv2d(64, 64,kernel_size=(2,2),stride=1, padding=0)
		self.relu = nn.ReLU()
		self.pooling1 = nn.AdaptiveAvgPool2d((4,4))
		self.pooling2 = nn.AdaptiveAvgPool2d((2,2))		
	
	def forward(self,ID5_timestep, ID4_timestep):
		features_ID5 = self.conv2dlayer1_ID5(ID5_timestep)
		features_ID5 = self.conv2dlayer2_ID5(features_ID5)
		features_ID5 = self.pooling2(features_ID5)
		
		features_ID4 = self.conv2dlayer1_ID4(ID4_timestep)
		features_ID4 = self.conv2dlayer2_ID4(features_ID4)
		features_ID4 = self.pooling2(features_ID4)

		concatenate = torch.cat((features_ID5, features_ID4), axis=1)
		concatenate = torch.squeeze(self.fclayer_cat(concatenate))
		
		return concatenate

class tactile_CNN(nn.Module):
	def __init__(self, tactile_length=2):
		super(tactile_CNN, self).__init__()
		self.CNN_blocks = CNN_block()

	def forward(self, ID5_timeseries, ID4_timeseries):
		timestep=0
		feature_dic = {}
		feature_list=[]			
		for i in 10 :
			feature_dic["timestep_"+str(timestep)] = self.CNN_blocks(ID5_timestep = ID5_timeseries[:,timestep,:,:,:], ID4_timestep = ID4_timeseries[:, timestep,:,:,:]) 
			feature_list.append(feature_dic["timestep_"+str(timestep)])
			timestep += 1 
		TCN_input_tactile = torch.cat((feature_list[0], feature_list[1]),axis=1)
		return TCN_input_tactile			


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

class vision_network(nn.Module):
	def __init__(self, base_network='vgg_16', pretrained=False, feature_extract=True):
		super(vision_network, self).__init__()
		self.features = None
		if base_network == 'vgg_16': 
			self.features = vgg16_bn(pretrained=pretrained)
			set_parameter_requires_grad(self.features, feature_extract)
			num_ftrs = self.features.classifier[6].in_features
			self.features.classifier[6] = nn.Linear(num_ftrs,128)
		if base_network == 'vgg_11':
			self.features = vgg11_bn(pretrained=pretrained)
			set_parameter_requires_grad(self.features, feature_extract)
			num_ftrs = self.features.classifier[6].in_features
			self.features.classifier[6] = nn.Linear(num_ftrs,128)
		elif base_network == 'resnet_18':
			self.features = resnet18(pretrained=pretrained)
			set_parameter_requires_grad(self.features, feature_extract)
			# To delete the last layer.
			self.features.fc = nn.Linear(512, 128)
		elif base_network == 'resnet_34':
			self.features = resnet34(pretrained=pretrained)
			set_parameter_requires_grad(self.features, feature_extract)
			# To delete the last layer.
			self.features.fc = nn.Linear(512, 128)
		elif base_network == 'resnet_50':
			self.features = resnet50(pretrained=pretrained)
			set_parameter_requires_grad(self.features, feature_extract)
			# To delete the last layer.
			self.features.fc = nn.Linear(2048, 128)
		elif base_network == 'wide_resnet_50':
			self.features = wide_resnet50_2(pretrained=pretrained)
			set_parameter_requires_grad(self.features, feature_extract)
			# To delete the last layer.
			self.features.fc = nn.Linear(2048, 128)
		elif base_network == 'inception_v3':
			self.features = inception_v3(pretrained=pretrained)
			set_parameter_requires_grad(self.features, feature_extract)
			# To delete the last layer.
			self.features.fc = nn.Linear(2048, 128)
		elif base_network == 'debug':
			self.features = alexnet(pretrained=pretrained)
			# To delete the last layer
			self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
			self.fc = nn.Sequential(nn.Linear(4096*2, 64))

		assert self.features, "Illegal CNN network name!"

	def forward(self, x):
	
		TCN_input_img = self.features(x)	
		if len(TCN_input_img) == 2:
			return TCN_input_img[0]
		else:
			return TCN_input_img
		
		return TCN_input_img


class vision_tactile_block(nn.Module):
	def __init__(self, base_network='resnet_50', pretrained=True, feature_extract=True, vision=True, tactile=True):
		super(vision_tactile_block, self).__init__()
		self.vision = vision
		self.tactile = tactile
		if self.vision == True and self.tactile==True:
			self.vision_block = vision_network(base_network=base_network, pretrained=pretrained, feature_extract=feature_extract)
			self.tactile_block = tactile_CNN(tactile_length=2)
			self.fc_vtcat = nn.Linear(256, 256)
			self.dropout = nn.Dropout(0.5)
		if self.vision == True and self.tactile==False:
			self.vision_block = vision_network(base_network=base_network, pretrained=pretrained, feature_extract=feature_extract)
		if self.vision == False and self.tactile==True:
			self.tactile_block = tactile_CNN(tactile_length=2)
	
	def forward(self, ID5_timeseries, ID4_timeseries, img_timestep):
		if self.vision == True and self.tactile==True:
			vision_features = self.vision_block(img_timestep)
			tactile_features = self.tactile_block(ID5_timeseries, ID4_timeseries)
			VT_features = torch.cat((vision_features, tactile_features), axis=1)
			VT_features = self.fc_vtcat(VT_features)
			VT_features = self.dropout(VT_features) 
		if self.vision == True and self.tactile==False:
			vision_features = self.vision_block(img_timestep) 
			VT_features = vision_features
		if self.vision == False and self.tactile==True:
			tactile_features = self.tactile_block(ID5_timeseries, ID4_timeseries)
			VT_features = tactile_features

		return VT_features

class tcn_single_block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dropout):
		super(tcn_single_block, self).__init__()
		self.tcn = TemporalConvNet_Single(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, dropout = dropout)

	def forward(self, x):
		output = self.tcn(x) 
		
		return output

class CNN_TCN(nn.Module):
	def __init__(self, in_channels=[[144,160]], out_channels=[[160,176]], kernel_size=[[3,5]], dropout=0.5, vision=True, tactile=True):
		super(CNN_TCN, self).__init__()
		self.VT_block = vision_tactile_block(base_network='inception_v3', pretrained=True, feature_extract=True, vision = vision, tactile=tactile)
		if vision == True and tactile==True:
			input_size = 256
		else:
			input_size = 128
		self.tcn = tcn_single_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dropout=dropout)
		self.fclayer1 = nn.Linear(out_channels[-1][-1], 2) 


	def forward(self, ID5_seq, ID4_seq, img_seq)
		timestep=0
		feature_dic = {}
		feature_list=[]
		length = img_seq.size()[1]
		for timestep in range(length):
			feature_dic["timestep_"+str(timestep)] = self.VT_block(ID5_timeseries = ID5_seq[:, timestep, :, :, :, :],  
																   ID4_timeseries = ID4_seq[:, timestep, :, :, :, :],
																   img_timestep = img_seq[:, timestep, :, :, :])

			feature_list.append(feature_dic["timestep_"+str(timestep)])
			timestep += 1 
		tcn_input=torch.stack(feature_list,axis=2)
		out = self.tcn(tcn_input) 
		concatenate = out[:,:,-1]

		prediction = self.fclayer1(concatenate)

		return prediction
	

