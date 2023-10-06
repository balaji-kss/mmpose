import json
import numpy as np
import pandas as pd
import pickle
import re
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

joint_order = [
    "nose", 
    "left_eye", 
    "right_eye", 
    "left_ear", 
    "right_ear", 
    "left_shoulder", 
    "right_shoulder", 
    "left_elbow", 
    "right_elbow", 
    "left_wrist", 
    "right_wrist", 
    "left_hip", 
    "right_hip", 
    "left_knee", 
    "right_knee", 
    "left_ankle", 
    "right_ankle" 
]

def get_wrnch_raw_data(file, frame_width, frame_height):
    '''Load Wrnch data as a nested python list'''
    filename = 'work_dirs/tumeke_testing/wrnch_jsons/{}.json'.format(file)
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    
    standard_format_data = []
    wrnch_to_hrnet = [16, 19, 17, 20, 18, 13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]
    for i in range(len(data["frames"])):
        wrnch_frame = data["frames"][i]
        single_frame = []
        for single_person_data in wrnch_frame["persons"]:
            if ("pose2d" not in single_person_data):
                continue
            if ("joints" not in single_person_data["pose2d"]):
                continue
            arr_2d = single_person_data["pose2d"]["joints"]
            if (arr_2d == []):
                continue
            kpts2d = np.array(arr_2d).reshape((25, 2))
            kpts2d[:, 0] *= frame_width
            kpts2d[:, 1] *= frame_height
            kpts2d = np.where(kpts2d < 0, 0, kpts2d)
            scores = np.array(single_person_data["pose2d"]["scores"])[wrnch_to_hrnet]
            standard_format = np.hstack((kpts2d[wrnch_to_hrnet], scores[:, None]))
            single_frame.append({
                "keypoints": standard_format
            })
        standard_format_data.append(single_frame)
    
    return standard_format_data


def get_pickle_joint_data(file):
    '''Pull pickled data'''
    # file_name = re.search('(.*)(?:[.])', file).group(1)
    with open (file, 'rb') as fp:
        raw_data = pickle.load(fp)

    '''Load picked data into numpy.

    Bounding box values are: 'xyxy' = (left, top, right, bottom)

    Key Point values are: (ndarray[Kx3]): x, y, score

    '''    
    return raw_data


def euclidean_dist(p1, p2):
    # Euclidean distance
    return np.linalg.norm(p1 - p2)


def closest_subject_heuristic(keypoint_array):
    '''Get avg euclidean distance between right shoulder/elbow, left shoulder/elbow, right shoulder/hip and left shoulder/hip''' 
    r_shoulder_i = joint_order.index('right_shoulder')
    r_elbow_i = joint_order.index('right_elbow')
    r_hip_i= joint_order.index('right_hip')
    l_shoulder_i = joint_order.index('left_shoulder')
    l_elbow_i = joint_order.index('left_elbow')
    l_hip_i = joint_order.index('left_hip')
    
    r_se_ed = euclidean_dist(keypoint_array[r_shoulder_i], keypoint_array[r_elbow_i])
    r_sh_ed = euclidean_dist(keypoint_array[r_shoulder_i], keypoint_array[r_hip_i])
    l_se_ed = euclidean_dist(keypoint_array[l_shoulder_i], keypoint_array[l_elbow_i])
    l_sh_ed = euclidean_dist(keypoint_array[l_shoulder_i], keypoint_array[l_hip_i])
    return np.array([r_se_ed, r_sh_ed, l_se_ed, l_sh_ed]).mean()


def pull_from_dict(key, dict):
    try:
        return dict[key]
    except (KeyError, TypeError):
        if key == 'bbox':
            return np.full(5,np.NaN)
        else:
            return np.NaN

        
def extract_subject(subject_num, subjects):
    try:
        return subjects[subject_num-1]
    except IndexError:
        return None 

    
def raw_data_to_dataframe(raw_data):
    """ Load into dataframe"""
    
    # Order subjects in each frame by bbox size
    sorted_raw_data = []
    for frame in raw_data:
        sorted_raw_data.append(sorted(frame, key=lambda x: closest_subject_heuristic(x['keypoints']), reverse=True))

    # Get first identified subject in every frame
    first_subj = np.array([extract_subject(1, i) for i in sorted_raw_data])

    # Extract relevant fields to be set as columns
    bbox = np.array([pull_from_dict('bbox', i) for i in first_subj])
    area = np.array([pull_from_dict('area', i) for i in first_subj])
    track_id = np.array([pull_from_dict('track_id', i) for i in first_subj])

    # Structure Keypoints
    keypoints_array = []
    for frame in first_subj:
        
        #TODO(znoland):sanity check this and make sure no issues!
        # Keypoint order! - configs/_base_/datasets/h36m.py
        if not frame:
            # If no data for subject
            frame_keypoints = {}
            for index in range(17):
                frame_keypoints['j{}_x'.format(index)] = np.NaN
                frame_keypoints['j{}_y'.format(index)] = np.NaN
                frame_keypoints['j{}_score'.format(index)] = np.NaN
                frame_keypoints['j{}_l'.format(index)] = joint_order[index]
            keypoints_array.append(frame_keypoints)
            
        else:
            # If data for subject
            frame_keypoints = {}
            for index, keypoint in enumerate(frame['keypoints']):
                frame_keypoints['j{}_x'.format(index)] = keypoint[0]
                frame_keypoints['j{}_y'.format(index)] = keypoint[1]
                frame_keypoints['j{}_score'.format(index)] = keypoint[2]
                frame_keypoints['j{}_l'.format(index)] = joint_order[index]
            keypoints_array.append(frame_keypoints)

    # Create seperate dataframes (otherwise can't combine as >1-dimensional fields in pandas or numpy)
    bbox_s = pd.DataFrame(bbox, columns=['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'bbox_score'])
    area_s = pd.DataFrame(area, columns=['area'])
    track_id_s = pd.DataFrame(track_id, columns=['track_id'])
    keypoints_s = pd.DataFrame(keypoints_array)


    # Create combined DataFrame
    df = pd.concat([bbox_s,area_s,track_id_s,keypoints_s], axis=1)
    
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'id'})
    return df

    #TODO(znoland): Set index as a frame number column (e.g. video1_frame_num)
    #TODO(znoland): Add column with the name of the video?


def get_hrnet_posenet_wrnch_dfs(file, width, height):
    '''Loads target file into data frames for further analysis'''
    
    hrnet_raw_data = get_pickle_joint_data(f'work_dirs/tumeke_testing/pickle_files/{file}.p')
    posenet_raw_data = get_pickle_joint_data(f'work_dirs/tumeke_testing/posenet_pickle_files/{file}.p')
    openpose_raw_data = get_pickle_joint_data(f'work_dirs/tumeke_testing/openpose_pickle_files/{file}.p')
    print('hrnet data loaded')
    wrnch_raw_data = get_wrnch_raw_data(file, width, height)
    print('wrnch data loaded')
    
    hrnet_df = raw_data_to_dataframe(hrnet_raw_data)
    print('hrnet data formatted')
    print('hrnet_df shape:', hrnet_df.shape)
    posenet_df = raw_data_to_dataframe(posenet_raw_data)
    print('posenet data formatted')
    print('posenet_df shape:', posenet_df.shape)
    wrnch_df = raw_data_to_dataframe(wrnch_raw_data)
    openpose_df = raw_data_to_dataframe(openpose_raw_data)
    print('wrnch data formatted')
    print('wrnch_df shape:', wrnch_df.shape)
    print('openpose data formatted')
    print('openpose_df shape:', openpose_df.shape)
    
    return hrnet_df, posenet_df, openpose_df, wrnch_df
    
    



