# Adapter 
"""The code to translate Argoverse dataset to KITTI dataset format"""

# Argoverse-to-KITTI Adapter

# Author: Yiyang Zhou 
# Email: yiyang.zhou@berkeley.edu

# Extension: Di Feng
# Email: di.feng@berkeley.edu

print('\nLoading files...')
import os
from shutil import copyfile
import json
import numpy as np
import math
from typing import Union
import pyntcloud
import progressbar
from itertools import chain
from time import sleep
from scipy.spatial.transform import Rotation as R
_PathLike = Union[str, "os.PathLike[str]"]

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST, get_image_dims_for_camera, STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT, RING_IMG_HEIGHT, RING_IMG_WIDTH
from argoverse.utils.calibration import CameraConfig
from argoverse.utils.cv2_plotting_utils import draw_clipped_line_segment
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat, quat_argo2scipy, quat_argo2scipy_vectorized
from argoverse.utils import calibration

import matplotlib.pyplot as plt
"""
Your original file directory is:
argodataset
└── argoverse-tracking <----------------------------root_dir
    └── train <-------------------------------------data_dir
        └── 0ef28d5c-ae34-370b-99e7-6709e1c4b929
        └── 00c561b9-2057-358d-82c6-5b06d76cebcf
        └── ...
    └── validation
        └──5c251c22-11b2-3278-835c-0cf3cdee3f44
        └──...
    └── test
        └──8a15674a-ae5c-38e2-bc4b-f4156d384072
        └──...

"""
####CONFIGURATION#################################################
# Root directory
root_dir = '/media/vision/HDD Storage/data/argoverse/argoverse-tracking/'
# Set up the data dir and target dir
data_dir_list = ['train1/', 'train2/', 'train3/', 'train4/', 'val/']
goal_dir = root_dir + 'argoverse-kitti/'
goal_subdir = goal_dir + 'training/'
imageset_dir = goal_dir+'/ImageSets' # to store train val mapping

if not os.path.exists(goal_dir): os.mkdir(goal_dir)
if not os.path.exists(goal_dir+'statistics'): os.mkdir(goal_dir+'statistics') # store dataset statistics

if not os.path.exists(imageset_dir): os.mkdir(imageset_dir) 
if not os.path.exists(goal_subdir): 
    os.mkdir(goal_subdir)
    os.mkdir(goal_subdir+'velodyne')
    os.mkdir(goal_subdir+'image_2')
    os.mkdir(goal_subdir+'calib')
    os.mkdir(goal_subdir+'label_2')
    os.mkdir(goal_subdir+'velodyne_semantics')
    

# Maximum thresholding distance for labelled objects
# (Object beyond this distance will not be labelled)
max_d = 70

# Camera reference setup
cams_all = ['ring_front_center',
 'ring_front_left',
 'ring_front_right',
 'ring_rear_left',
 'ring_rear_right',
 'ring_side_left',
 'ring_side_right'] #All available cameras
cam_id = 0 # Choose only one of camera as reference. TODO: save labels for all cameras
cam = cams_all[cam_id]

# Sample rate to avoid sequential data (original 10Hz)
sample_rate = 1

# remove ground lidar points when generating /velodyne
remove_ground = False

# Map info
create_map = False # When set True, create drivable road maps and ground maps as rasterized maps with 1x1 meters
map_x_limit = [-40, 40] # lateral distance
map_y_limit = [0, 70] # longitudinal distance
raster_size = 1.0 # argoverse map resolution (meter)

create_map_semantics = False # When set True, create "velodyne_semantics" to append lidar data with map information

if create_map or create_map_semantics or remove_ground:
    from argoverse.map_representation.map_api import ArgoverseMap
    argoverse_map = ArgoverseMap()

# Official categorization from the paper "Train in Germany, Test in The USA: Making 3D Object Detectors Generalize", from Xiangyu Chen et al. CVPR2020
# Information provided by Xiangyu Chen by courtesy xc429@cornell.edu
CLASS_MAP = {
    "VEHICLE": "Car",
    "PEDESTRIAN": "Pedestrian",
    "LARGE_VEHICLE": "Truck",
    "BICYCLIST": "Cyclist",
    "BUS": "Truck",
    "TRAILER": "Truck",
    "MOTORCYCLIST": "Cyclist",
    "EMERGENCY_VEHICLE": "Van",
    "SCHOOL_BUS": "Truck"
}


# save data conversion configuration
config = '=========== Argo2KITTI conversion configuration ========== \n' + \
        'sample_rate = {}'.format(sample_rate) + '\n' + \
        'max_d = {}'.format(max_d) + '\n' + \
        'selected_camera = ' + cams_all[cam_id] + '\n' + \
        'remove_ground = {}'.format(remove_ground) + '\n' + \
        'create_map = {}'.format(create_map) + '\n' + \
        'create_map_semantics = {}'.format(create_map_semantics) + '\n' + \
        '==========================================================='
print(config)
with open(os.path.join(goal_dir, 'config.txt'),'a') as f:
    f.write(config)
    f.close()
####################################################################


def adapter():
    # calculate dataset statistics
    object_statistics = {'Car':[],'Pedestrian':[],'Cyclist':[], 'Truck':[], 'Van':[]} #[w, l, h, bottom_z (camera coordinate), center_z (lidar coordinate)]                
    map_statistics = {'plane': []} # [plane_z_min, plane_z_max, plane_z_mean, plane_z_var], all in lidar coordinate

    # kitti data index 
    kitti_idx = 0 
    for dr in data_dir_list:
        data_dir = root_dir + dr
        # Check the number of logs(one continuous trajectory)
        argoverse_loader= ArgoverseTrackingLoader(data_dir)
        print('\nConvert file: ', dr)
        print('\nTotal number of logs:',len(argoverse_loader))
        argoverse_loader.print_all()
        print('\n')

        #cams = cams_all if cam_id<0 else [cams_all[cam_id]]

        # count total number of files
        total_number=0
        for q in argoverse_loader.log_list:
            path, dirs, files = next(os.walk(data_dir+q+'/lidar'))
            total_number= total_number+len(files)

        #total_number = total_number*7 if cam_id<0 else total_number

        #bar = progressbar.ProgressBar(maxval=total_number, \
        #    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

        #print('Total number of files: {}. Translation starts...'.format(total_number))
        #print('Progress:')
        #bar.start()

        for log_id_n, log_id in enumerate(argoverse_loader.log_list):
            argoverse_data= argoverse_loader.get(log_id)
            city_name = argoverse_data.city_name
            
            if create_map or create_map_semantics or remove_ground: 
                ground_height_mat, npyimage_to_city_se2_mat = argoverse_map.get_rasterized_ground_height(city_name) # map information of the city

            #for cam in cams: 

            # Recreate the calibration file content 
            calibration_data = argoverse_data.get_calibration(cam)
            extrinsic= calibration_data.extrinsic
            ext_rot= R.from_matrix(extrinsic[0:3,0:3].T)
            trans= -(extrinsic[0:3,3]).reshape(3,1)
            extrinsic_kitti= np.hstack((extrinsic[0:3,0:3],-trans))

            L3='P2: '
            for j in calibration_data.K.reshape(1,12)[0]:
                L3= L3+ str(j)+ ' '
            L3=L3[:-1]

            L6= 'Tr_velo_to_cam: '
            for k in extrinsic_kitti.reshape(1,12)[0][0:12]:
                L6= L6+ str(k)+ ' '
            L6=L6[:-1]


            L1='P0: 0 0 0 0 0 0 0 0 0 0 0 0'
            L2='P1: 0 0 0 0 0 0 0 0 0 0 0 0'
            L4='P3: 0 0 0 0 0 0 0 0 0 0 0 0'
            L5='R0_rect: 1 0 0 0 1 0 0 0 1'
            L7='Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0'

            file_content="""{}
{}
{}
{}
{}
{}
{}
     """.format(L1,L2,L3,L4,L5,L6,L7)

            # Loop through the each lidar frame (10Hz) to copy and reconfigure all images, lidars, calibration files, and label files.  
            lidar_timestamp_list = argoverse_data.lidar_timestamp_list
            frame_idx_list = range(len(lidar_timestamp_list))

            for frame_idx in frame_idx_list[::sample_rate]:
                # Save lidar file into .bin format under the new directory 
                target_lidar_file_path = goal_subdir + 'velodyne/'+ str(kitti_idx).zfill(6) + '.bin'
                lidar_data = argoverse_data.get_lidar(frame_idx)
                lidar_data_augmented = np.concatenate((lidar_data,np.zeros([lidar_data.shape[0],1])),axis=1) # intensity
                lidar_data_augmented = lidar_data_augmented.astype('float32')
                lidar_data_augmented.tofile(target_lidar_file_path)

                # Save the image file into .png format under the new directory 
                cam_file_path = argoverse_data.image_list_sync[cam][frame_idx]
                target_cam_file_path = goal_subdir +'image_2/' + str(kitti_idx).zfill(6) + '.png'
                copyfile(cam_file_path,target_cam_file_path)

                file=open(goal_subdir+'calib/' + str(kitti_idx).zfill(6) + '.txt','w+')
                file.write(file_content)
                file.close()

                label_object_list = argoverse_data.get_label_object(frame_idx)
                file=open(goal_subdir +  'label_2/' + str(kitti_idx).zfill(6) + '.txt','w+')

                # For map information
                if create_map: 
                    # first generate meshgrid 
                    city_coord_x, city_coord_y, _ = argoverse_data.get_pose(frame_index).translation # ego vehicle pose in the city coordinate
                    city_coord_x_range = np.arange(city_coord_x + map_x_limit[0], city_coord_x + map_x_limit[1] + 1, raster_size)
                    city_coord_y_range = np.arange(city_coord_y + map_y_limit[0], city_coord_y + map_y_limit[1] + 1, raster_size)
                    xx, yy = np.meshgrid(city_coord_x_range, city_coord_y_range)
                    xv, yv = xx.flatten(), yy.flatten()
                    city_coords = np.array([xv,yv]).T
                    city_coords = np.round(city_coords).astype(np.int64)

                    # then generate height info
                    zv = argoverse_map.get_ground_height_at_xy(city_coords, city_name)
                    # we do not care the ground height for non-drivable areas
                    drivable_area_bool = argoverse_map.get_raster_layer_points_boolean(city_coords, city_name, "driveable_area")
                    non_drivable_area_bool = ~drivable_area_bool
                    zv[non_drivable_area_bool] = -1000 # set as invalid values

                    ground_height_map = zv.reshape([city_coord_y_range.shape[0], city_coord_x_range.shape[0]]) # in the city coordinate
                    drivable_binary_map = drivable_area_bool.reshape([city_coord_y_range.shape[0], city_coord_x_range.shape[0]])

                if create_map_semantics or remove_ground:
                    # only drivable area has lidar ground! in the drivable areas, a lidar point may belong to ground (nor object) 
                    
                    '''
                    TODO: more lidar semantics (such as point cloud class)

                                        #Values    Name      Description
                    ----------------------------------------------------------------------------
                       1    ground_height         ground height value for each lidar point. Note
                                                  that only drivable areas has valid ground height.
                                                  invalid heights are set to be -1000
                       1    lidar_ground_bool     bool, whether lidar point belongs to ground,
                                                  threshold set 0.3m according to Argoverse
                       1    drivable_area_bool    bool, whether lidar point belongs to a drivable
                                                  area.
                    ''' 
                    #TODO: consider to append features directly to lidar.bin file, instead of creating new files
                    city_to_egovehicle_se3 = argoverse_data.get_pose(frame_idx) # city coordinate transformer
                    lidar_data_city_coords = city_to_egovehicle_se3.transform_point_cloud(lidar_data) # ego-vehilce coordinate to city coordinate
                    # binary mapping
                    lidar_ground_bool = argoverse_map.get_ground_points_boolean(lidar_data_city_coords, city_name)
                    drivable_area_bool = argoverse_map.get_raster_layer_points_boolean(lidar_data_city_coords, city_name, "driveable_area")
                    
                    if create_map_semantics:    
                        target_lidar_semantics_file_path = goal_subdir + 'velodyne_semantics/'+ str(kitti_idx).zfill(6) + '.bin'    
                        # ground height
                        ground_heights_city_coords = argoverse_map.get_ground_height_at_xy(lidar_data_city_coords, city_name) 
                        ground_data_city_coords = np.stack([lidar_data_city_coords[:,0], lidar_data_city_coords[:,1], ground_heights_city_coords], axis=1)
                        ground_data_ego_coords = city_to_egovehicle_se3.inverse_transform_point_cloud(ground_data_city_coords[drivable_area_bool])
                        ground_heights = -1000 * np.ones(lidar_ground_bool.shape[0]) # only drivable area has ground heights, other areas are set to be 1000!
            
                        i = 0
                        for j, v in enumerate(drivable_area_bool):
                            if v: 
                                ground_heights[j] = ground_data_ego_coords[i,2]
                                i += 1

                        lidar_semantics = np.stack([ground_heights, lidar_ground_bool, drivable_area_bool], axis=1).astype('float32')
                        lidar_semantics.tofile(target_lidar_semantics_file_path)

                        map_statistics['plane'].append([ground_heights[drivable_area_bool].min(), ground_heights[drivable_area_bool].max(), 
                            ground_heights[drivable_area_bool].mean(), ground_heights[drivable_area_bool].var()])

                    if remove_ground:
                        lidar_data_filtered = lidar_data_augmented[lidar_ground_bool==0, :]
                        lidar_data_filtered.tofile(target_lidar_file_path) # overwrite the original lidar data


                # For each object label
                has_object = False
                for detected_object in label_object_list:
                    if detected_object.label_class not in CLASS_MAP.keys(): continue # skip this object of non-interest class
                    classes = CLASS_MAP[detected_object.label_class]
                    quaternion = detected_object.quaternion # w, x, y, z
                    quat = R.from_quat(quaternion)
                    occulusion = round(detected_object.occlusion/25) #TODO: need to improve, as 25 is only valid for kitti but not for argoverse
                
                    height = detected_object.height
                    length = detected_object.length
                    width = detected_object.width
                    center = detected_object.translation # in ego frame, [x,y,z] with with x forward, y left, and z up

                    corners_ego_frame=detected_object.as_3d_bbox() # all eight points in ego frame 
                    corners_cam_frame= calibration_data.project_ego_to_cam(corners_ego_frame) # all eight points in the camera frame 
                    image_corners= calibration_data.project_ego_to_image(corners_ego_frame)
                    center_cam_frame= calibration_data.project_ego_to_cam(np.array([center]))
                    # the four coordinates we need for KITTI
                    image_bbox = list(chain(np.min(image_corners, axis=0).tolist()[0:2], np.max(image_corners, axis=0).tolist()[0:2])) 
                    
                    # calculate truncation, 
                    # codes following the paper "Train in Germany, Test in The USA: Making 3D Object Detectors Generalize"
                    # author: xc429@cornell.edu
                    inside = (0 <= image_bbox[1] < RING_IMG_HEIGHT and 0 < image_bbox[3] <= RING_IMG_HEIGHT) and (
                        0 <= image_bbox[0] < RING_IMG_WIDTH and 0 < image_bbox[2] <= RING_IMG_WIDTH) and np.min(corners_cam_frame[:, 2], axis=0) > 0
                    valid = (0 <= image_bbox[1] < RING_IMG_HEIGHT or 0 < image_bbox[3] <= RING_IMG_HEIGHT) and (
                        0 <= image_bbox[0] < RING_IMG_WIDTH or 0 < image_bbox[2] <= RING_IMG_WIDTH) and np.min(corners_cam_frame[:, 2], axis=0) > 0 and detected_object.translation[0] > 0
                    #

                    label_keep = 0<center_cam_frame[0][2]<max_d 

                    # only keep labels in image FOV and in valid distance range 
                    if valid and label_keep:
                        has_object = True
                        truncated = valid and not inside
                        if truncated:
                            _bbox = [0] * 4
                            _bbox[0] = max(0, image_bbox[0])
                            _bbox[1] = max(0, image_bbox[1])
                            _bbox[2] = min(RING_IMG_WIDTH, image_bbox[2])
                            _bbox[3] = min(RING_IMG_HEIGHT, image_bbox[3])
                            truncated = 1.0 - ((_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])) / ((image_bbox[2] - image_bbox[0]) * (image_bbox[3] - image_bbox[1]))
                            image_bbox = _bbox
                        else:
                            truncated = 0.0
                        ### 

                        # the center coordinates in cam frame we need for KITTI 
                        # for the orientation, we choose point 1 and point 5 for application 
                        p1= corners_cam_frame[1]
                        p5= corners_cam_frame[5]
                        dz=p1[2]-p5[2]
                        dx=p1[0]-p5[0]

                        # the orientation angle of the car and the observation angle alpha
                        # codes following the paper "Train in Germany, Test in The USA: Making 3D Object Detectors Generalize"
                        # author: xc429@cornell.edu
                        dcm_LiDAR = argoverse.utils.transform.quat2rotmat(quaternion) 
                        #!!Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format, 
                        #whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
                        #two formats here. We use the [w, x, y, z] order because this corresponds to the
                        #multidimensional complex number `w + ix + jy + kz`.
                        dcm_cam = calibration_data.R.dot(dcm_LiDAR.dot(calibration_data.R.T))
                        rot_y = -np.pi * 0.5 + R.from_matrix(dcm_cam).as_rotvec()[1]
                        angle = np.arctan2(np.sin(rot_y), np.cos(rot_y))

                        alpha = -np.arctan2(center_cam_frame[0, 0], center_cam_frame[0, 2]) + rot_y
                        # 

                        tr_x = center_cam_frame[0][0] # x lateral (right)
                        tr_y = center_cam_frame[0][1] + height*0.5 # y vertical (down)
                        tr_z = center_cam_frame[0][2] # z longitudinal (forward)

                        '''
                                            #Values    Name      Description
                        ----------------------------------------------------------------------------
                           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                             'Misc' or 'DontCare'
                           1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                                             truncated refers to the object leaving image boundaries
                           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                                             0 = fully visible, 1 = partly occluded
                                             2 = largely occluded, 3 = unknown
                           1    alpha        Observation angle of object, ranging [-pi..pi]
                           4    bbox         2D bounding box of object in the image (0-based index):
                                             contains left, top, right, bottom pixel coordinates
                           3    dimensions   3D object dimensions: height, width, length (in meters)
                           3    location     3D object location x,y,z in camera coordinates (in meters)
                           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
                           1    score        Only for results: Float, indicating confidence in
                                             detection, needed for p/r curves, higher is better.
                        '''
                        line=classes+ ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(
                            round(truncated,2),
                            occulusion,
                            round(alpha,2),
                            round(image_bbox[0],2),
                            round(image_bbox[1],2),
                            round(image_bbox[2],2),
                            round(image_bbox[3],2),
                            round(height,2), 
                            round(width,2),
                            round(length,2), 
                            round(tr_x,2), 
                            round(tr_y,2), 
                            round(tr_z,2), 
                            round(angle,2))                

                        file.write(line)

                        object_statistics[classes].append([width, length, height, tr_y, center[2]])

                file.close()

                # store index txt
                if dr == 'val/' and has_object: 
                    file=open(imageset_dir + '/val.txt','a')
                    file.write(str(kitti_idx).zfill(6)+' \n')
                    file.close()
                elif dr == 'test/' and has_object:
                    file=open(imageset_dir + '/test.txt','a')
                    file.write(str(kitti_idx).zfill(6)+' \n')
                    file.close()
                elif has_object: # training
                    file=open(imageset_dir + '/train.txt','a')
                    file.write(str(kitti_idx).zfill(6)+' \n')
                    file.close()

                kitti_idx+= 1
                
                #if kitti_idx< total_number:
                #    bar.update(kitti_idx+1)

                #print('kitti_idx = ',str(kitti_idx),' log_id = ',log_id_n,' frame_idx',frame_idx, ' log_id = ', log_id)
            #bar.finish()
    return object_statistics, map_statistics

def subset_mapping():
    """
    create subsample of the argoverse dataset
    """
    assert os.path.isdir(imageset_dir) 

    train_id_list = [x.strip() for x in open(os.path.join(imageset_dir, 'train.txt')).readlines()]
    val_id_list = [x.strip() for x in open(os.path.join(imageset_dir, 'val.txt')).readlines()]

    train_id_list_sub20 = train_id_list[::5]
    val_id_list_sub20 = val_id_list[::5]

    train_id_list_sub10 = train_id_list[::10]
    val_id_list_sub10 = train_id_list[::10]

    if os.path.isfile(os.path.join(imageset_dir, 'train20.txt')): 
        os.remove(os.path.join(imageset_dir, 'train20.txt'))

    if os.path.isfile(os.path.join(imageset_dir, 'val20.txt')): 
        os.remove(os.path.join(imageset_dir, 'val20.txt'))

    if os.path.isfile(os.path.join(imageset_dir, 'train10.txt')): 
        os.remove(os.path.join(imageset_dir, 'train10.txt'))

    if os.path.isfile(os.path.join(imageset_dir, 'val10.txt')): 
        os.remove(os.path.join(imageset_dir, 'val10.txt'))

    with open(os.path.join(imageset_dir, 'train20.txt'),'a') as f:
        for d in train_id_list_sub20: f.write(d+' \n')
        f.close()

    with open(os.path.join(imageset_dir, 'val20.txt'),'a') as f:
        for d in val_id_list_sub20: f.write(d+' \n')
        f.close()

    with open(os.path.join(imageset_dir, 'train10.txt'),'a') as f:
        for d in train_id_list_sub10: f.write(d+' \n')
        f.close()

    with open(os.path.join(imageset_dir, 'val10.txt'),'a') as f:
        for d in val_id_list_sub20: f.write(d+' \n')
        f.close()

def print_statistics(object_statistics, map_statistics):
    with open(goal_dir+'statistics/statistics.txt','a') as f:
        for classes in object_statistics.keys():
            data = np.array(object_statistics[classes])
            minimal, maximal, mean, variance, num = data.min(axis=0), data.max(axis=0), data.mean(axis=0), data.var(axis=0), data.shape[0]
            f.write('============= ' + classes + ' statistics  ============= \n')
            f.write('number of objects = ' + str(num) +'\n')
            f.write('Width min: ' + str(round(minimal[0],3)) + ' max: ' + str(round(maximal[0],3)) + ' mean: ' + str(round(mean[0],3)) + ' var: ' + str(round(variance[0],3)) + '\n')
            f.write('Length min: ' + str(round(minimal[1],3)) + ' max: ' + str(round(maximal[1],3)) + ' mean: ' + str(round(mean[1],3)) + ' var: ' + str(round(variance[1],3)) + '\n')
            f.write('Heigth min: ' + str(round(minimal[2],3)) + ' max: ' + str(round(maximal[2],3)) + ' mean: ' + str(round(mean[2],3)) + ' var: ' + str(round(variance[2],3)) + '\n')
            f.write('Bottom_z min: ' + str(round(minimal[3],3)) + ' max: ' + str(round(maximal[3],3)) + ' mean: ' + str(round(mean[3],3)) + ' var: ' + str(round(variance[3],3)) + '\n')
            f.write('Center_z min: ' + str(round(minimal[4],3)) + ' max: ' + str(round(maximal[4],3)) + ' mean: ' + str(round(mean[4],3)) + ' var: ' + str(round(variance[4],3)) + '\n')
            
            print('============= ' + classes + ' statistics  ============= ')
            print('number of objects = ', num)
            print('Width min: ', round(minimal[0],3), ' max: ', round(maximal[0],3), ' mean: ', round(mean[0],3), ' var: ', round(variance[0],3))
            print('Length min: ', round(minimal[1],3), ' max: ', round(maximal[1],3), ' mean: ', round(mean[1],3), ' var: ', round(variance[1],3))
            print('Heigth min: ', round(minimal[2],3), ' max: ', round(maximal[2],3), ' mean: ', round(mean[2],3), ' var: ', round(variance[2],3))
            print('Bottom_z min: ', round(minimal[3],3), ' max: ', round(maximal[3],3), ' mean: ', round(mean[3],3), ' var: ', round(variance[3],3))
            print('Center_z min: ', round(minimal[4],3), ' max: ', round(maximal[4],3), ' mean: ', round(mean[4],3), ' var: ', round(variance[4],3))
            print('\n')
        
            plt.figure(figsize=(15,3))
            plt.subplot(151)
            plt.hist(data[:,0], 20, density=True, facecolor='r', alpha=0.75)
            plt.xlabel('Width')
            plt.ylabel('Probability')
            plt.title('Mean = ' + str(round(mean[0],2)))
            plt.tight_layout()

            plt.subplot(152)
            plt.hist(data[:,1], 20, density=True, facecolor='g', alpha=0.75)
            plt.xlabel('Length')
            plt.ylabel('Probability')
            plt.title('Mean = ' + str(round(mean[1],2)))
            plt.tight_layout()

            plt.subplot(153)
            plt.hist(data[:,2], 20, density=True, facecolor='b', alpha=0.75)
            plt.xlabel('Heigth')
            plt.ylabel('Probability')
            plt.title('Mean = ' + str(round(mean[2],2)))
            plt.tight_layout()

            plt.subplot(154)
            plt.hist(data[:,3], 20, density=True, facecolor='m', alpha=0.75)
            plt.xlabel('Bottom_z (camera frame)')
            plt.ylabel('Probability')
            plt.title('Mean = ' + str(round(mean[3],2)))
            plt.tight_layout()

            plt.subplot(155)
            plt.hist(data[:,4], 20, density=True, facecolor='orange', alpha=0.75)
            plt.xlabel('Bottom_z (ego frame')
            plt.ylabel('Probability')
            plt.title('Mean = ' + str(round(mean[4],2)))
            plt.tight_layout()

            plt.savefig(goal_dir+'statistics/'+classes+'.pdf')
            plt.clf()


        plane = map_statistics['plane']
        if len(plane):
            plane = np.array(plane)
            plain_data = plane.mean(axis=0)
            f.write('=============  plane statistics  =============')
            f.write('Plane min: ' + str(round(plain_data[0],3)) + ' max: '+ str(round(plain_data[1],3)) + ' mean: ' + str(round(plain_data[2], 3)) + ' var: ' + str(round(plain_data[3], 3)) + '\n')
            print('=============  plane statistics  =============')
            print('Plane min: ', round(plain_data[0],3), ' max: ', round(plain_data[1],3), ' mean: ', round(plain_data[2], 3), ' var: ', round(plain_data[3], 3))

if __name__ == "__main__":
    object_statistics, map_statistics = adapter()
    subset_mapping()
    print_statistics(object_statistics, map_statistics)