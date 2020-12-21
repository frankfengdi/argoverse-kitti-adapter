# Adapter 
"""The code to translate Argoverse dataset to KITTI dataset format"""


# Argoverse-to-KITTI Adapter

# Author: Yiyang Zhou 
# Email: yiyang.zhou@berkeley.edu

print('\nLoading files...')

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import os
from shutil import copyfile
from argoverse.utils import calibration
import json
import numpy as np
from argoverse.utils.calibration import CameraConfig
from argoverse.utils.cv2_plotting_utils import draw_clipped_line_segment
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
import math
from typing import Union
import pyntcloud
import progressbar
from time import sleep
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

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
root_dir= '/media/vision/HDD Storage/data/argoverse/argoverse-tracking/'

# Maximum thresholding distance for labelled objects
# (Object beyond this distance will not be labelled)
max_d=70

# When set False, labels out of camera frames are ignored, 
# Otherwise contain all label 
need_full_label=False 

# All available cameras
cams_all = ['ring_front_center',
 'ring_front_left',
 'ring_front_right',
 'ring_rear_left',
 'ring_rear_right',
 'ring_side_left',
 'ring_side_right']

# When set -1, store data from all cameras, otherwise choose only of them from cams_all
cam_id = 0

# Sample rate to avoid sequential data (original 10Hz)
sample_rate = 5

# Set up the data dir and target dir
data_dir_list = ['train1/', 'train2/', 'train3/', 'train4/', 'val/']
goal_dir = root_dir + 'argoverse-kitti/'
goal_subdir = goal_dir + 'training/'

# When set True, create drivable road maps and ground maps as rasterized maps with 1x1 meters
create_map = False
map_x_limit = [-40, 40] # lateral distance
map_y_limit = [0, 70] # longitudinal distance
raster_size = 1.0 # argoverse map resolution (meter)
if create_map:
    from argoverse.map_representation.map_api import ArgoverseMap
    argoverse_map = ArgoverseMap()

####################################################################
if not os.path.exists(goal_dir): os.mkdir(goal_dir)
if not os.path.exists(goal_subdir): 
    os.mkdir(goal_subdir)
    os.mkdir(goal_subdir+'velodyne')
    os.mkdir(goal_subdir+'image_2')
    os.mkdir(goal_subdir+'calib')
    os.mkdir(goal_subdir+'label_2')
    os.mkdir(goal_subdir+'velodyne_reduced')

_PathLike = Union[str, "os.PathLike[str]"]

i = 0 # kitti data index 
for dr in data_dir_list:
    data_dir = root_dir + dr
    # Check the number of logs(one continuous trajectory)
    argoverse_loader= ArgoverseTrackingLoader(data_dir)
    print('\nConvert file: ', dr)
    print('\nTotal number of logs:',len(argoverse_loader))
    argoverse_loader.print_all()
    print('\n')

    cams = cams_all if cam_id<0 else [cams_all[cam_id]]

    # count total number of files
    total_number=0
    for q in argoverse_loader.log_list:
        path, dirs, files = next(os.walk(data_dir+q+'/lidar'))
        total_number= total_number+len(files)

    total_number = total_number*7 if cam_id<0 else total_number

    bar = progressbar.ProgressBar(maxval=total_number, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    print('Total number of files: {}. Translation starts...'.format(total_number))
    print('Progress:')
    bar.start()

    for log_id_n, log_id in enumerate(argoverse_loader.log_list):
        argoverse_data= argoverse_loader.get(log_id)
        city_name = argoverse_data.city_name
        
        if create_map: 
            ground_height_mat, npyimage_to_city_se2_mat = argoverse_map.get_rasterized_ground_height(city_name) # map information of the city

        for cam in cams:
            # Recreate the calibration file content 
            calibration_data=calibration.load_calib(data_dir+log_id+'/vehicle_calibration_info.json')[cam]
            
            extrinsic= calibration_data.extrinsic
            ext_rot= R.from_matrix(extrinsic[0:3,0:3].T)
            trans= -(extrinsic[0:3,3]).reshape(3,1)
            extrinsic_kitti= np.hstack((extrinsic[0:3,0:3],-trans))

            #print(extrinsic)
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
            l=0

            # Loop through the each lidar frame (10Hz) to copy and reconfigure all images, lidars, calibration files, and label files.  
            lidar_timestamp_list = argoverse_data.lidar_timestamp_list
            frame_idx_list = range(len(lidar_timestamp_list))

            for frame_idx in frame_idx_list[::sample_rate]:
                # Save lidar file into .bin format under the new directory 
                target_lidar_file_path = goal_subdir + 'velodyne/'+ str(i).zfill(6) + '.bin'

                lidar_data = argoverse_data.get_lidar(frame_idx)
                lidar_data_augmented = np.concatenate((lidar_data,np.zeros([lidar_data.shape[0],1])),axis=1) # intensity
                lidar_data_augmented = lidar_data_augmented.astype('float32')
                lidar_data_augmented.tofile(target_lidar_file_path)

                # Save the image file into .png format under the new directory 
                cam_file_path = argoverse_data.image_list_sync[cam][frame_idx]
                target_cam_file_path = goal_subdir +'image_2/' + str(i).zfill(6) + '.png'
                copyfile(cam_file_path,target_cam_file_path)

                file=open(goal_subdir+'calib/' + str(i).zfill(6) + '.txt','w+')
                file.write(file_content)
                file.close()

                label_object_list = argoverse_data.get_label_object(frame_idx)
                file=open(goal_subdir +  'label_2/' + str(i).zfill(6) + '.txt','w+')

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
                    drivable_area_bool = am.get_raster_layer_points_boolean(city_coords, city_name, "driveable_area")
                    non_drivable_area_bool = ~drivable_area_bool
                    zv[non_drivable_area_bool] = -1000 # set as invalid values

                    ground_height_map = zv.reshape([city_coord_y_range.shape[0], city_coord_x_range.shape[0]]) # in the city coordinate
                    drivable_binary_map = drivable_area_bool.reshape([city_coord_y_range.shape[0], city_coord_x_range.shape[0]])

                # For each object label
                has_object = False
                for detected_object in label_object_list:
                    quat= Quar = R.from_quat(detected_object.quaternion)
                    classes = detected_object.label_class
                    occulusion = round(detected_object.occlusion/25)
                    height = detected_object.height
                    length = detected_object.length
                    width = detected_object.width
                    center= detected_object.translation # in ego frame, [x,y,z]
                    quaternion = detected_object.quaternion #rot_w, rot_x, rot_y, rot_z

                    truncated= 0

                    corners_ego_frame=detected_object.as_3d_bbox() # all eight points in ego frame 
                    corners_cam_frame= calibration_data.project_ego_to_cam(corners_ego_frame) # all eight points in the camera frame 
                    image_corners= calibration_data.project_ego_to_image(corners_ego_frame)
                    image_bbox= [min(image_corners[:,0]), min(image_corners[:,1]),max(image_corners[:,0]),max(image_corners[:,1])]
                    # the four coordinates we need for KITTI
                    image_bbox=[round(x) for x in image_bbox]      

                    center_cam_frame= calibration_data.project_ego_to_cam(np.array([center]))

                    # flag to set bboxes out of image FOV ignored 
                    label_keep = 0<center_cam_frame[0][2]<max_d and 0<image_bbox[0]<1920 and 0<image_bbox[1]<1200 and 0<image_bbox[2]<1920 and 0<image_bbox[3]<1200

                    if not need_full_label and label_keep or need_full_label:
                        has_object = True
                        # the center coordinates in cam frame we need for KITTI 
                        # for the orientation, we choose point 1 and point 5 for application 
                        p1= corners_cam_frame[1]
                        p5= corners_cam_frame[5]
                        dz=p1[2]-p5[2]
                        dx=p1[0]-p5[0]
                        # the orientation angle of the car
                        angle= ext_rot * quat
                        angle=angle.as_euler('zyx')[1]
                        angle=-1*angle
                        angle = (angle + np.pi) % (2 * np.pi) - np.pi 
                        beta= math.atan2(center_cam_frame[0][2],center_cam_frame[0][0])
                        alpha= (angle-beta + np.pi) % (2 * np.pi) - np.pi 

                        #alpha = angle + beta - math.pi/2
                        tr_x = center_cam_frame[0][0] # x lateral
                        tr_y = center_cam_frame[0][1] + height*0.5 # y vertical (negative)
                        tr_z = center_cam_frame[0][2] # z longitudinal


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

                file.close()


                # store index txt
                if dr == 'val/' and has_object: 
                    file=open(goal_subdir+'val.txt','a')
                    file.write(str(i).zfill(6)+' \n')
                    file.close()
                elif dr == 'test/' and has_object:
                    file=open(goal_subdir+'test.txt','a')
                    file.write(str(i).zfill(6)+' \n')
                    file.close()
                elif has_object: # training
                    file=open(goal_subdir+'train.txt','a')
                    file.write(str(i).zfill(6)+' \n')
                    file.close()

                i+= 1
                if i< total_number:
                    bar.update(i+1)

                #print('i = ',str(i),' log_id = ',log_id_n,' frame_idx',frame_idx, ' log_id = ', log_id)


        bar.finish()

print('Translation finished, processed {} files'.format(i))
