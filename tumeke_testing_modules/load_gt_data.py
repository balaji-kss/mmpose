import re
import numpy as np
import pandas as pd
import json
from tumeke_testing_modules.load_model_data import euclidean_dist

joint_order = [
    'Nose', 
    'L_Eye', 
    'R_Eye', 
    'L_Ear', 
    'R_Ear', 
    'L_Shoulder', 
    'R_Shoulder', 
    'L_Elbow', 
    'R_Elbow', 
    'L_Wrist', 
    'R_Wrist', 
    'L_Hip', 
    'R_Hip', 
    'L_Knee', 
    'R_Knee', 
    'L_Ankle', 
    'R_Ankle' 
]

def extract_x_and_y(joint):
    return np.array([joint['x'], joint['y']])


def closest_subject_heuristic_raw(keypoint_obj):
    '''Get avg euclidean distance between right shoulder/elbow, left shoulder/elbow, right shoulder/hip and left shoulder/hip''' 
    r_shoulder_i = joint_order.index('R_Shoulder')
    r_elbow_i = joint_order.index('R_Elbow')
    r_hip_i= joint_order.index('R_Hip')
    l_shoulder_i = joint_order.index('L_Shoulder')
    l_elbow_i = joint_order.index('L_Elbow')
    l_hip_i = joint_order.index('L_Hip')
    
    r_se_ed = euclidean_dist(extract_x_and_y(keypoint_obj[r_shoulder_i]), extract_x_and_y(keypoint_obj[r_elbow_i]))
    r_sh_ed = euclidean_dist(extract_x_and_y(keypoint_obj[r_shoulder_i]), extract_x_and_y(keypoint_obj[r_hip_i]))
    l_se_ed = euclidean_dist(extract_x_and_y(keypoint_obj[l_shoulder_i]), extract_x_and_y(keypoint_obj[l_elbow_i]))
    l_sh_ed = euclidean_dist(extract_x_and_y(keypoint_obj[l_shoulder_i]), extract_x_and_y(keypoint_obj[l_hip_i]))
    return np.array([r_se_ed, r_sh_ed, l_se_ed, l_sh_ed]).mean()


def ground_truth_processing(labels, multiple_subjects):
    '''Tranform ground truth data into data frame

        1) Get Subjects

        2) Get Keypoints for each subject

        3) Order subjects by heuristic

    '''

    frame_array = []

    for label in labels:
    #     label = label_r[0]
        # label = labels[0:1][0]

        row = {}

        # Metadata
        img_name = re.search(r'(\d\d+)', label['data']['img']).group(0)
        row['id'] = int(img_name)
        row['ls_id'] = label['id']
        row['img'] = label['data']['img']
        row['subjects'] = []

        #Subjects
        persons_t = [p for p in label['annotations'][0]['result'] if p['type'] == 'rectanglelabels']

        # Keypoints
        keypoints_t = [k for k in label['annotations'][0]['result'] if k['type'] == 'keypointlabels']

        for i, person in enumerate(persons_t):

            # Filter on subject's keypoints
            if multiple_subjects:
                person_keypoints_t = [k for k in keypoints_t if k['parentID'] == person['id']]
            else: 
                person_keypoints_t = keypoints_t
                
            # Filter out unused keypoints
            person_keypoints_t = [k for k in person_keypoints_t if k['value']['keypointlabels'][0] in joint_order]

             # Set order of keypoints + set placeholders when null
            person_keypoints_sorted = []
            for i, joint_name in enumerate(joint_order):
                target_keypoint = [k for k in person_keypoints_t if k['value']['keypointlabels'][0] == joint_name]

                if target_keypoint:
                    original_width = target_keypoint[0]['original_width']
                    original_height = target_keypoint[0]['original_height']
                    target_keypoint = target_keypoint[0]['value']
                    target_keypoint['original_width'] = original_width
                    target_keypoint['original_height'] = original_height
                else:
                    target_keypoint = {'x': np.nan, 'y': np.nan, 'width': np.nan, 'keypointlabels': [joint_name], 'original_width': np.nan, 'original_height': np.nan, }

                person_keypoints_sorted.append(target_keypoint)

            subject_t = {
                'name': person['value']['rectanglelabels'][0],
                'id': person['id'],
                'pose-keypoints': person_keypoints_sorted,
                'closest-subject-heuristic': closest_subject_heuristic_raw(person_keypoints_sorted)
            }
            row['subjects'].append(subject_t)


        # Sort subjects by heuristic
        row['subjects'] = sorted(row['subjects'], key=lambda x: x['closest-subject-heuristic'], reverse=True)

        # Format x & y coordinates (for 1st subject only)
        for index, keypoint in enumerate(row['subjects'][0]['pose-keypoints']):
            if len(keypoint['keypointlabels']) > 1:
                raise ValueError('There are multiple labels for one keypoint!') 
            row['j{}_x'.format(index)] = keypoint['x'] / 100.0 * keypoint['original_width']
            row['j{}_y'.format(index)] = keypoint['y'] / 100.0 * keypoint['original_height']
            row['j{}_l'.format(index)] = keypoint['keypointlabels'][0]

        frame_array.append(row)
        
    return frame_array


def get_gt_df(file, multiple_subjects):
    '''Load joint file from label studio and processes the data'''
    with open('work_dirs/tumeke_testing/ground_truth_labels/{}.json'.format(file)) as f:
        labels = json.load(f)
        
    # raw array is "frame_array" !
    gt_frame_array = ground_truth_processing(labels, multiple_subjects)
    gt_df = pd.DataFrame(gt_frame_array)
    
    # Order by ID
    gt_df = gt_df.sort_values(by='id')
    gt_df = gt_df.reset_index(drop=True)
    gt_df = gt_df.set_index('id')
    print('ground truth loaded and formatted')
    print('gt_df shape:', gt_df.shape)
    
    return gt_df



