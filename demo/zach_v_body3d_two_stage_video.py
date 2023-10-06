# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from argparse import ArgumentParser

import time
import cv2
import mmcv
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

from mmpose.core.visualization import imshow_bboxes, imshow_keypoints
from mmpose.apis import (extract_pose_sequence, get_track_id,
                         inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model, init_pose_model_trt,
                         process_mmdet_results, vis_3d_pose_result)
from mmdet2trt.apis import create_wrap_detector

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
def handpose_keypoint_convert(hands_2d_l, hands_2d_r):
    return np.vstack((hands_2d_l, hands_2d_r))

def covert_keypoint_definition(keypoints, pose_det_dataset, pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition.
    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.
    """
    if pose_det_dataset == 'TopDownH36MDataset' and \
    (pose_lift_dataset == 'Body3DH36MDataset' or pose_lift_dataset == 'Body3DH36MModifiedDataset'):
        return keypoints
    elif ((pose_det_dataset == 'TopDownCocoDataset') and \
            (pose_lift_dataset == 'Body3DH36MDataset' or pose_lift_dataset == 'Body3DH36MModifiedDataset')):
        keypoints_new = np.zeros((17, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # head is in the middle of l_eye and r_eye
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new
    elif (pose_det_dataset == 'TopDownCocoWholeBodyDataset' and \
          pose_lift_dataset == 'Body3DCombinedDataset'):
        keypoints_new = np.zeros((17, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # head is in the middle of l_eye and r_eye
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new
    elif (pose_det_dataset == 'TopDownCocoDataset') and \
            (pose_lift_dataset == 'Body3DCombinedDataset'):
        
        keypoints_new = np.zeros((19, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[17] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[18] = (keypoints[5] + keypoints[6]) / 2
        
        keypoints_new[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        return keypoints_new
    else:
        raise NotImplementedError

def create_writer(frame, args, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        osp.join(args.out_video_root, f'tumeke_testing/vis_{osp.basename(args.file_path)}_hands'),
        fourcc,
        fps,
        (frame.shape[1], frame.shape[0])
    )
    return writer

def process_video_hands(args):
    assert has_mmdet, 'Please install mmdet to run the demo.'

#     args = parser.parse_args()
    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    video = mmcv.VideoReader(args.file_path)
    assert video.opened, f'Failed to load video file {args.file_path}'

    # First stage: 2D pose detection
    print('Stage #1: 2D pose detection.')

    person_det_model = create_wrap_detector(
        "/home/fsuser/PoseEstimation/mmpose/faster_rcnn.trt", 
        "/home/fsuser/PoseEstimation/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        "cuda:0"
    )

    pose_det_model_trt = init_pose_model_trt(
        "hrnet.onnx",
        "hrnet.trt",
        args.pose_detector_config,
        "2947",
        device=args.device.lower())
    print("Initialized Model")
    assert pose_det_model_trt.cfg.model.type == 'TopDown', 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

#     print("pose_det_dataset->", pose_det_model.cfg.data['test']['type'])
    pose_det_dataset = pose_det_model_trt.cfg.data['test']['type']
    pose_det_results_list = []
    next_id = 0
    pose_det_results = []
    idx = 0
    if (args.detections_2d == ""):
        print(f"Video len: {len(video)}")
        for frame in video:
            t1 = time.time()
            pose_det_results_last = pose_det_results
            
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(person_det_model, frame)

            # keep the person class bounding boxes.
            person_det_results = process_mmdet_results(mmdet_results,
                                                       args.det_cat_id)
            # make person results for single image
            # cv2.imwrite("test.jpg", frame)
            pose_det_results, _ = inference_top_down_pose_model(
                pose_det_model_trt,
                frame,
                person_det_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=pose_det_dataset,
                return_heatmap=False,
                outputs=None,
                trt=True)

            # get track id for each person instance
            pose_det_results, next_id = get_track_id(
                pose_det_results,
                pose_det_results_last,
                next_id,
                use_oks=args.use_oks_tracking,
                tracking_thr=args.tracking_thr,
                use_one_euro=args.euro,
                fps=video.fps)
            idx += 1
            print(f"Time: {time.time() - t1}")
            if (idx % 100 == 0): print(f"Idx: {idx}")
            pose_det_results_list.append(copy.deepcopy(pose_det_results))
        # Pickle keypoints
        with open(f'work_dirs/tumeke_testing/pickle_files/{args.video_name}.p', 'wb') as outfile:
            pickle.dump(pose_det_results_list, outfile)
    else:
        with open(args.detections_2d, 'rb') as f:
            pose_det_results_list = pickle.load(f)
        
    # Second stage: Pose lifting
    print('Stage 2: 2D-to-3D pose lifting.')

    pose_lift_model = init_pose_model_trt(
        "videopose.onnx",
        "videopose.trt",
        args.pose_lifter_config,
        "116",
        device=args.device.lower())

    assert pose_lift_model.cfg.model.type == 'PoseLifter', \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    
#     print("pose_lift_dataset->", pose_lift_model.cfg.data['test']['type'])
    
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = covert_keypoint_definition(
                keypoints, pose_det_dataset, pose_lift_dataset)
    # load temporal padding config from model.data_cfg
    if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
        data_cfg = pose_lift_model.cfg.test_data_cfg
    else:
        data_cfg = pose_lift_model.cfg.data_cfg
    
    if (isinstance(data_cfg, list)):
        data_cfg = data_cfg[0]

    num_instances = args.num_instances
    for i, pose_det_results in enumerate(
            mmcv.track_iter_progress(pose_det_results_list)):
        # extract and pad input pose2d sequence
        
                
        pose_results_2d = extract_pose_sequence(
            pose_det_results_list,
            frame_idx=i,
            causal=data_cfg.causal,
            seq_len=data_cfg.seq_len,
            step=3)
        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=pose_results_2d,
            dataset=pose_lift_dataset,
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d,
            dataset_info=pose_lift_model.cfg.dataset_info,
            output_num=i,
            trt=True)
        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            
            keypoints_3d = res['keypoints_3d']
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # add title
            det_res = pose_det_results[idx]
            instance_id = det_res['track_id']
            res['title'] = f'Prediction ({instance_id})'
            # only visualize the target frame
            res['keypoints'] = det_res['keypoints']
            if ("bbox" in det_res):
                res['bbox'] = det_res['bbox']
            else:
                res['bboxes'] = det_res['bboxes']
            res['track_id'] = instance_id
            pose_lift_results_vis.append(res)

        # Visualization
        if num_instances < 0:
            num_instances = len(pose_lift_results_vis)
        img_vis = vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=video[i],
            out_file=None,
            radius=args.radius,
            thickness=args.thickness,
            num_instances=num_instances)

        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    osp.join(args.out_video_root,
                             f'tumeke_testing/vis_{osp.basename(args.file_path)}'), fourcc,
                    fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)

    if save_out_video:
        writer.release()
    
    print(f'Video "{args.video_name}" processed.')


if __name__ == '__main__':
    process_video(args)
