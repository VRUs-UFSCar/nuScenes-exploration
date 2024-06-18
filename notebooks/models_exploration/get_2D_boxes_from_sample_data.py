'''
Todo código deste arquivo foi adaptado de:
https://github.com/asvath/mobile_robotics/blob/master/nuscenes%20extract%20and%20write%20out%202d%20annotation%20boxes-revised%20to%20truncate%20bb.ipynb
'''


import numpy as np
from nuscenes.nuscenes import NuScenes

from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility


def threeD_2_twoD(boxsy, intrinsic):
    '''
    Transforma uma caixa 3D em uma caixa 2D
    '''
    corners = boxsy.corners()
    x = corners[0,:]
    y = corners[1,:]
    z = corners[2,:]
    x_y_z = np.array((x,y,z))
    orthographic = np.dot(intrinsic,x_y_z)
    perspective_x = orthographic[0]/orthographic[2]
    perspective_y = orthographic[1]/orthographic[2]
    
    min_x = np.min(perspective_x)
    max_x = np.max(perspective_x)
    min_y = np.min(perspective_y)
    max_y = np.max(perspective_y)
    
    return min_x, min_y, max_x, max_y

def get_2D_boxes_from_sample_data(nusc_object: NuScenes, sample_data_token: str, box_vis_level=BoxVisibility.ANY, selected_anntokens=None):
    """
    Recebe um objeto da base de dados NuScenes e um token de sample_data.

    Retorna uma lista de caixas 2D e as categorias (classes/labels) das anotações associadas a essas caixas.

    As lista de caixas seguirá o seguinte formato:
    [ (min_x, min_y, max_x, max_y), ...]

    As categorias seguirão o seguinte formato:
    [ 'category1', 'category2', ...]
    """

    # Retrieve sensor & pose records
    sd_record = nusc_object.get('sample_data', sample_data_token)
    cs_record = nusc_object.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc_object.get('sensor', cs_record['sensor_token'])
    pose_record = nusc_object.get('ego_pose', sd_record['ego_pose_token'])

    sample_record = nusc_object.get('sample',sd_record['sample_token'])

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc_object.get_box, selected_anntokens))
    else:
        boxes = nusc_object.get_boxes(sample_data_token)
        selected_anntokens = sample_record['anns']

    box_list = []
    ann_list = []
    for box, ann in zip(boxes, selected_anntokens):

        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        
        # Append 2D box to list
        box_list.append(threeD_2_twoD(box, cam_intrinsic))

        # Append annotation category to list
        ann_info = nusc_object.get('sample_annotation', ann)
        ann_instance = nusc_object.get('instance', ann_info['instance_token'])
        ann_category = nusc_object.get('category', ann_instance['category_token'])
        ann_list.append(ann_category['name'])

    return box_list, ann_list