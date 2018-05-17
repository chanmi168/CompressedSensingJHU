
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from read_depth import depth_read
import os, sys

class RgbdDataset(Dataset):

	def __init__(self, rgb_dir, sparse_depth_dir, continuous_depth_dir, ground_dir, boundary_dir=None):
		self.identity = [x.replace('rgb.png', '') for x in sorted(os.listdir(rgb_dir))]
		self.rgb_list = [rgb_dir+x for x in sorted(os.listdir(rgb_dir))]
		self.sparse_depth_list = [sparse_depth_dir+x for x in sorted(os.listdir(sparse_depth_dir))]
		self.continuous_depth_list = [continuous_depth_dir+x for x in sorted(os.listdir(continuous_depth_dir))]
		self.ground_list = [ground_dir+x for x in sorted(os.listdir(ground_dir))]
		self.boundary_list = None
		if boundary_dir != None:
			self.boundary_list = [boundary_dir+x for x in sorted(os.listdir(boundary_dir))]
		

	def __len__(self):
		return len(self.rgb_list)

	def __getitem__(self, idx):
		rgb_data = cv2.imread(self.rgb_list[idx])
		#sparse_depth_data = cv2.imread(self.sparse_depth_list[idx], 0)
		#continuous_depth_data = cv2.imread(self.continuous_depth_list[idx], 0)
		#ground_data = cv2.imread(self.ground_list[idx], 0)
		sparse_depth_data = depth_read(self.sparse_depth_list[idx])
		continuous_depth_data = depth_read(self.continuous_depth_list[idx])
		ground_data = depth_read(self.ground_list[idx])
		identity = self.identity[idx]
		return (rgb_data, sparse_depth_data, continuous_depth_data, ground_data, identity)
