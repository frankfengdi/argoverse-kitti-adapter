import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import yaml
import pandas as pd
from kitti_util import *
from pylab import *
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

goal_dir = '/media/vision/HDD Storage/data/argoverse/argoverse-tracking/argoverse-kitti/training'
imageset_dir = os.path.join(goal_dir, '..', 'ImageSets')
lidar_dir = os.path.join(goal_dir, 'velodyne')
ground_dir = os.path.join(goal_dir, 'velodyne_semantics')
if not os.path.exists('image_plane'): os.mkdir('image_plane')

train_id_list = [x.strip() for x in open(os.path.join(imageset_dir, 'train.txt')).readlines()]
for train_id in train_id_list[:20]:
	print(train_id)
	lidar_data = np.fromfile(os.path.join(lidar_dir, train_id + '.bin'), dtype=np.float32).reshape(-1,4) # x,y,z,intensity
	semantic_data = np.fromfile(os.path.join(ground_dir, train_id + '.bin'), dtype=np.float32).reshape(-1,3) # ground_height, lidar_ground_bool,drivable_area_bool
	
	plt.figure(figsize=(15,6))
	gs = gridspec.GridSpec(5, 15)
	ax = plt.subplot(gs[:4, :5])
	plt.scatter(lidar_data[:,0], lidar_data[:,1], marker='.',  c='blue', alpha=0.7, s=0.1)
	plt.title('Full LiDAR points on BEV')
	plt.xlim([-80,80])
	plt.ylim([-80,80])
	plt.tight_layout()

	ax = plt.subplot(gs[:4, 5:10])
	plt.scatter(lidar_data[semantic_data[:,2]==1,0], lidar_data[semantic_data[:,2]==1,1], marker='.',  c='red', alpha=0.7, s=0.1, label='drivable area')
	plt.scatter(lidar_data[semantic_data[:,2]==0,0], lidar_data[semantic_data[:,2]==0,1], marker='.',  c='blue', alpha=0.7, s=0.1, label='non drivable area')
	plt.title('Drivable LiDAR points on BEV')
	plt.legend()
	plt.xlim([-80,80])
	plt.ylim([-80,80])
	plt.tight_layout()

	ax = plt.subplot(gs[:4, 10:15])
	plt.scatter(lidar_data[semantic_data[:,1]==1,0], lidar_data[semantic_data[:,1]==1,1], marker='.',  c='green', alpha=0.7, s=0.1, label='ground')
	plt.scatter(lidar_data[semantic_data[:,1]==0,0], lidar_data[semantic_data[:,1]==0,1], marker='.',  c='blue', alpha=0.7, s=0.1, label='non ground')
	plt.title('Ground LiDAR points on BEV')
	plt.legend()
	plt.xlim([-80,80])
	plt.ylim([-80,80])
	plt.tight_layout()

	ax = plt.subplot(gs[4,:15])
	plt.scatter(lidar_data[semantic_data[:,2]==1,1], lidar_data[semantic_data[:,2]==1,2], marker='.',  c='blue', alpha=0.7, s=0.1, label='original pcl')
	plt.scatter(lidar_data[semantic_data[:,2]==1,1], semantic_data[semantic_data[:,2]==1,0], marker='.',  c='m', alpha=0.7, s=0.1, label='ground pcl')
	plt.title('Orignal pcl and corr. ground')
	plt.legend()
	plt.xlim([-40,40])
	plt.ylim([-5, 5])
	plt.tight_layout()

	plt.savefig('image_plane/'+train_id+'.png', dpi=150)
	plt.clf()