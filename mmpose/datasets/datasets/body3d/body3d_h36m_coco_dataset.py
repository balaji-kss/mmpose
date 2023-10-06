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
from ...builder import DATASETS

@DATASETS.register_module(name="Body3DH36MCOCODataset")
class Body3DH36MCOCODataset(Kpt3dSviewKpt2dDataset):
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

    COCO_JOINT_NAMES = [
        'Nose',
        'L_Eye',
        'R_Eye',
        'L_Ear',
        'R_Ear', \
        'L_Shoulder',
        'R_Shoulder',
        'L_Elbow',
        'R_Elbow', \
        'L_Wrist',
        'R_Wrist',
        'L_Hip',
        'R_Hip',
        'L_Knee', \
        'R_Knee', \
        'L_Ankle',
        'R_Ankle',
        "Pelvis",
        "Head"
    ]

    H36M_TO_COCO = [
        -1, -1, -1, -1, -1, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3, 0, 10
    ]

    # 2D joint source options:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate by the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
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
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # h36m specific attributes
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}"' + \
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(
                data_cfg['camera_param_file'])

        # h36m specific annotation info
        ann_info = {}
        ann_info['use_different_joint_weights'] = False
        # action filter
        actions = data_cfg.get('actions', '_all_')
        self.actions = set(
            actions if isinstance(actions, (list, tuple)) else [actions])

        # subject filter
        subjects = data_cfg.get('subjects', '_all_')
        self.subjects = set(
            subjects if isinstance(subjects, (list, tuple)) else [subjects])

        self.ann_info.update(ann_info)

    @staticmethod
    def process_bbox(bbox, width, height):
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if w*h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1])
        else:
            return None

        # aspect ratio preserving bbox
        w = bbox[2]
        h = bbox[3]
        c_x = bbox[0] + w/2.
        c_y = bbox[1] + h/2.
        aspect_ratio = 1
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bbox[2] = w*1.25
        bbox[3] = h*1.25
        bbox[0] = c_x - bbox[2]/2.
        bbox[1] = c_y - bbox[3]/2.
        return bbox

    def _get_subject(self):
        if not self.test_mode:
            if self.protocol == 1:
                subject = [1,5,6,7,8,9]
            elif self.protocol == 2:
                subject = [1,5,6,7,8]
        else:
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
       
        return subject

    def _get_subsampling_ratio(self):
        if not self.test_mode:
            return 5
        else:
            return 5
    
    @staticmethod
    def _cam2pixel(cam_coord, f, c):
        x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
        y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
        z = cam_coord[:, 2]
        img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
        return img_coord

    @staticmethod
    def _pixel2cam(pixel_coord, f, c):
        x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
        y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
        z = pixel_coord[:, 2]
        cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
        return cam_coord
    
    @staticmethod 
    def _world2cam(world_coord, R, t):
        cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
        return cam_coord

    def _transform_coords(self, joint_coord):
        transformed_coords = np.zeros((19, joint_coord.shape[1]))
        transformed_coords = joint_coord[self.H36M_TO_COCO]
        transformed_coords[[0, 1, 2, 3, 4]] = np.zeros((1, joint_coord.shape[1]))
        
        return transformed_coords

    def load_annotations(self):
        """
            Reads 3DMPPE Posenet H36M annotations,
            returns them in the following format: 

            data_info = {
                'imgnames': _imgnames,
                'joints_3d': _joints_3d,
                'joints_2d': _joints_2d,
                'scales': _scales,
                'centers': _centers,
            }
        """
        # get 2D joints
        self.ann_prefix = self.ann_file
        subject_list = self._get_subject()
        sampling_ratio = self._get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}

        data_info = {
            'imgnames': [],
            'joints_3d': [],
            'joints_2d': [],
            'scales': [],
            'centers': [],
        }

        for subject in subject_list:
            # data load
            with open(osp.join(self.ann_prefix, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.ann_prefix, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.ann_prefix, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
        db.createIndex()
        print("Get bounding box and root from groundtruth")
        data = []


        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_prefix, img['file_name'])
            img_width, img_height = img['width'], img['height']
           
            # check subject and frame_idx
            subject = img['subject']; frame_idx = img['frame_idx'];
            if subject not in subject_list:
                continue
            if frame_idx % sampling_ratio != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            # project world coordinate to cam, image coordinate space
            action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_world = self._transform_coords(joint_world)
            joint_cam = Body3DH36MCOCODataset._world2cam(joint_world, R, t)
            joint_img = Body3DH36MCOCODataset._cam2pixel(joint_cam, f, c)
            joint_img[:,2] = joint_img[:,2] - joint_cam[self.root_idx,2]
            joint_vis = np.ones((self.joint_num,1))
            joint_vis[[0, 1, 2, 3, 4]] = 0

            bbox = Body3DH36MCOCODataset.process_bbox(np.array(ann['bbox']), img_width, img_height)
            if bbox is None: continue
            root_cam = joint_cam[self.root_idx]

            data_info["imgnames"].append(img['file_name'])
            data_info["joints_3d"].append(np.hstack((joint_cam, joint_vis)))
            data_info["joints_2d"].append(np.hstack((joint_img[:, :2], joint_vis)))
            data_info["scales"].append(max(bbox[2], bbox[3]))
            center = joint_img[17, :2]
            data_info["centers"].append(center)
        data_info["joints_3d"] = np.array(data_info["joints_3d"])/1000
        data_info["joints_2d"] = np.array(data_info["joints_2d"])
        data_info["scales"] = np.array(data_info["scales"])
        data_info["centers"] = np.array(data_info["centers"])
        data_info["imgnames"] = np.array(data_info["imgnames"])
        return data_info

    @staticmethod
    def _parse_h36m_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        A typical h36m modified image filename is like:
        s_11_act_16_subact_02_ca_04_001360.jpg
        """
        name_parts = osp.basename(imgname).split('_')
        return name_parts[1], name_parts[3]+"_"+name_parts[5], name_parts[7]

    def build_sample_indices(self):
        """Split original videos into sequences and build frame indices.

        This method overrides the default one in the base class.
        """

        # Group frames into videos. Assume that self.data_info is
        # chronological.


        video_frames = defaultdict(list)
        for idx, imgname in enumerate(self.data_info['imgnames']):
            subj, action, camera = self._parse_h36m_imgname(imgname)

            if '_all_' not in self.actions and action not in self.actions:
                continue

            if '_all_' not in self.subjects and subj not in self.subjects:
                continue

            video_frames[(subj, action, camera)].append(idx)

        # build sample indices
        sample_indices = []
        _len = (self.seq_len - 1) * self.seq_frame_interval + 1
        _step = self.seq_frame_interval
        for _, _indices in sorted(video_frames.items()):
            n_frame = len(_indices)

            if self.temporal_padding:
                # Pad the sequence so that every frame in the sequence will be
                # predicted.
                if self.causal:
                    frames_left = self.seq_len - 1
                    frames_right = 0
                else:
                    frames_left = (self.seq_len - 1) // 2
                    frames_right = frames_left
                for i in range(n_frame):
                    pad_left = max(0, frames_left - i // _step)
                    pad_right = max(0,
                                    frames_right - (n_frame - 1 - i) // _step)
                    start = max(i % _step, i - frames_left * _step)
                    end = min(n_frame - (n_frame - 1 - i) % _step,
                              i + frames_right * _step + 1)
                    sample_indices.append([_indices[0]] * pad_left +
                                          _indices[start:end:_step] +
                                          [_indices[-1]] * pad_right)
            else:
                seqs_from_video = [
                    _indices[i:(i + _len):_step]
                    for i in range(0, n_frame - _len + 1)
                ]
                sample_indices.extend(seqs_from_video)

        # reduce dataset size if self.subset < 1
        assert 0 < self.subset <= 1
        subset_size = int(len(sample_indices) * self.subset)
        start = np.random.randint(0, len(sample_indices) - subset_size + 1)
        end = start + subset_size

        return sample_indices[start:end]

    def evaluate(self,
                 outputs,
                 res_folder,
                 metric='mpjpe',
                 logger=None,
                 **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        res_file = osp.join(res_folder, 'result_keypoints.json')
        kpts = []
        for output in outputs:
            preds = output['preds']
            image_paths = output['target_image_paths']
            batch_size = len(image_paths)
            for i in range(batch_size):
                if (image_paths[i] not in self.name2id):
                    continue
                target_id = self.name2id[image_paths[i]]
                
                kpts.append({
                    'keypoints': preds[i],
                    'target_id': target_id,
                })

        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(kpts)
            elif _metric == 'p-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='p-mpjpe')
            elif _metric == 'n-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='n-mpjpe')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        return OrderedDict(name_value_tuples)

    def _report_mpjpe(self, keypoint_results, mode='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants like
        P-MPJPE or N-MPJPE.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DH36MDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:
                - ``'mpjpe'``: Standard MPJPE.
                - ``'p-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
                - ``'n-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    in scale only.
        """

        preds = []
        gts = []
        masks = []
        action_category_indices = defaultdict(list)
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            target_id = result['target_id']
            gt, gt_visible = np.split(
                self.data_info['joints_3d'][target_id], [3], axis=-1)
            preds.append(pred)
            gts.append(gt)
            masks.append(gt_visible)
            action = self._parse_h36m_imgname(
                self.data_info['imgnames'][target_id])[1]
            action_category = action.split('_')[0]
            action_category_indices[action_category].append(idx)

        preds = np.stack(preds)
        gts = np.stack(gts)
        masks = np.stack(masks).squeeze(-1) > 0
        err_name = mode.upper()
        if mode == 'mpjpe':
            alignment = 'none'
        elif mode == 'p-mpjpe':
            alignment = 'procrustes'
        elif mode == 'n-mpjpe':
            alignment = 'scale'
        else:
            raise ValueError(f'Invalid mode: {mode}')
        error, preds = keypoint_mpjpe(preds, gts, masks, alignment)
        np.save("preds.npy", preds)
        np.save("gts.npy", gts)
        name_value_tuples = [(err_name, error)]

        for action_category, indices in action_category_indices.items():
            _error, _preds = keypoint_mpjpe(preds[indices], gts[indices],
                                    masks[indices])
            name_value_tuples.append((f'{err_name}_{action_category}', _error))

        return name_value_tuples

    def _load_camera_param(self, camera_param_file):
        """Load camera parameters from file."""
        camera_params = {}
        subject_list = self._get_subject()
        for subject in subject_list:
            with open(osp.join(camera_param_file, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                data = json.load(f)
                for camera_str in list(data.keys()):
                    cam = int(camera_str)
                    camera_params[(f"{subject:02d}", f"{cam:02d}")] = {
                        "R": data[camera_str]["R"],
                        "T": data[camera_str]["t"],
                        "c": data[camera_str]["c"],
                        "f": data[camera_str]["f"],
                        'w': 1000,
                        'h': 1002
                    }

        return camera_params

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        subj, _, camera = self._parse_h36m_imgname(imgname)

        return self.camera_param[(subj, camera)]
