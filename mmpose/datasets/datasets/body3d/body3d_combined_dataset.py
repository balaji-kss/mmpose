# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import json
import warnings
from collections import OrderedDict, defaultdict
from pycocotools.coco import COCO

import mmcv
import numpy as np
from mmcv import Config

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.datasets.datasets.base import Kpt3dSviewKpt2dDataset
from mmpose.datasets.datasets.body3d import Body3DH36MModifiedDataset
from ...builder import DATASETS

@DATASETS.register_module(name="Body3DCombinedDataset")
class Body3DCombinedDataset(Kpt3dSviewKpt2dDataset):
    """Human3.6M dataset for 3D human pose estimation.

    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Human3.6M keypoint indexes::
        0: 'root (pelvis)',
        1: 'right_hip',
        2: 'right_knee',
        3: 'right_foot',
        4: 'left_hip',
        5: 'left_knee',
        6: 'left_foot',
        7: 'spine',
        8: 'thorax',
        9: 'neck_base',
        10: 'head',
        11: 'left_shoulder',
        12: 'left_elbow',
        13: 'left_wrist',
        14: 'right_shoulder',
        15: 'right_elbow',
        16: 'right_wrist'


    Args:
        ann_prefix (str): Path to the annotation files folder.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    JOINT_NAMES = [
        'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
        'Thorax', 'NeckBase', 'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist'
    ]

    # 2D joint source options:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate by the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
                 child_types,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/h36m.py')
            dataset_info = cfg._cfg_dict['dataset_info']
        
        self.protocol = 2
        self.lshoulder_idx = 11
        self.rshoulder_idx = 14
        self.thorax_idx = 8
        self.root_idx = 0
        self.joint_num = 19
        self.old_to_new_coords = [0, 1, 2, 3, 4, 5, 6, 7, 0, 8, 10, 11, 12, 13, 14, 15, 16]

        self.child_datasets = []
        for i in range(len(child_types)):
            obj_cls = DATASETS.get(child_types[i])
            self.child_datasets.append(
                obj_cls(
                    ann_file[i],
                    img_prefix[i],
                    data_cfg[i],
                    pipeline,
                    dataset_info=dataset_info,
                    test_mode=test_mode

                )
            )
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)


    def load_config(self, data_cfg_list):
        self.seq_len = self.child_datasets[0].seq_len
        self.causal = self.child_datasets[0].causal
        self.num_joints = self.child_datasets[0].num_joints
        self.seq_frame_interval = self.child_datasets[0].seq_frame_interval 
        self.temporal_padding = self.child_datasets[0].temporal_padding
        self.subset = self.child_datasets[0].subset
        self.need_2d_label = self.child_datasets[0].need_2d_label
        self.need_camera_param = False

    def load_annotations(self):
        self.lens = []
        data_info = {
            'imgnames': [],
            'joints_3d': [],
            'joints_2d': [],
            'scales': [],
            'centers': [],
        }

        for i in range(len(self.child_datasets)):
            ret = self.child_datasets[i].data_info
            self.lens.append(len(ret['imgnames']))
            data_info['imgnames'] += ret['imgnames'].tolist()
            data_info['joints_3d'] += ret['joints_3d'].tolist()
            data_info['joints_2d'] += ret['joints_2d'].tolist()
            data_info['scales'] += ret['scales'].tolist()
            data_info['centers'] += ret['centers'].tolist()

        data_info["joints_3d"] = np.array(data_info["joints_3d"]).astype(np.float32)
        data_info["joints_2d"] = np.array(data_info["joints_2d"]).astype(np.float32)
        data_info["scales"] = np.array(data_info["scales"]).astype(np.float32)
        data_info["centers"] = np.array(data_info["centers"]).astype(np.float32)
        data_info["imgnames"] = np.array(data_info["imgnames"])
        print(f'Final len: {(data_info["joints_3d"].shape)}')
        return data_info

    def build_sample_indices(self):
        sample_indices = []
        for i in range(len(self.child_datasets)):
            ret = self.child_datasets[i].sample_indices
            ret = np.array(ret)
            if (i > 0):
                ret += self.lens[i-1]
            sample_indices += ret.tolist()
        return sample_indices

    def evaluate(self,
                 outputs,
                 res_folder,
                 metric='mpjpe',
                 logger=None,
                 **kwargs):

        return self.child_datasets[0].evaluate(outputs, res_folder, metric, logger, **kwargs)

    # Make cleaner
    def get_camera_param(self, imgname):
        for i in range(len(self.child_datasets)):
            ret = None
            try:
                ret = self.child_datasets[i].get_camera_param(imgname)
            except:
                continue
            return ret

