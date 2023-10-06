# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
import yaml
import glob
#from dex_ycb_toolkit.factory import get_dataset
from collections import OrderedDict, defaultdict

import numpy as np
import mmcv
from mmcv import Config

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.datasets.builder import DATASETS
from ..base import Kpt3dSviewKpt2dDataset


@DATASETS.register_module()
class DexYCBDataset(Kpt3dSviewKpt2dDataset):
    """DexYCB dataset for 2D-to-3D pose lifting

    DexYCB keypoint indexes::
        0: 'wrist',
        1: 'thumb_mcp',
        2: 'thumb_pip',
        3: 'thumb_dip',
        4: 'thumb_tip',
        5: 'index_mcp',
        6: 'index_pip',
        7: 'index_dip',
        8: 'index_tip',
        9: 'middle_mcp',
        10: 'middle_pip',
        11: 'middle_dip',
        12: 'middle_tip',
        13: 'ring_mcp',
        14: 'ring_pip',
        15: 'ring_dip',
        16: 'ring_tip',
        17: 'little_mcp',
        18: 'little_pip',
        19: 'little_dip',
        20: 'little_tip'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """
    
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}
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
            cfg = Config.fromfile('configs/_base_/datasets/dexycb.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        self.protocol = 2
        self.root_idx = 0
        self.joint_num = 42
        self._dex_ycb_dir = os.environ['DEX_YCB_DIR']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
        
        
        
    def _transform_coords(self, joint_cam):
        # SPINE is average of thorax and pelvis
        head = (joint_cam[1] + joint_cam[2] + joint_cam[3] + joint_cam[4])/4.0
        transformed_coords = joint_cam[self.AIST_TO_H36M]
        thorax = (transformed_coords[self.H36M_LSHOULDER_IDX] + transformed_coords[self.H36M_RSHOULDER_IDX])/2.0
        pelvis = (transformed_coords[self.H36M_LHIP_IDX] + transformed_coords[self.H36M_RHIP_IDX])/2.0

        spine = (thorax + pelvis)/2.0
        transformed_coords[self.H36M_SPINE_IDX] = spine
        transformed_coords[self.H36M_THORAX_IDX] = thorax
        transformed_coords[self.H36M_HEAD_IDX] = head
        return transformed_coords
    
    def left_or_right(self, joint_2d, joint_3d, side, vis):
        joint_2d_aug = np.zeros((42, 2))
        joint_3d_aug = np.zeros((42, 3))
        valid = np.zeros(42)
        if (side == "right"):
            joint_2d_aug[0:21, :] = joint_2d
            joint_3d_aug[0:21, :] = joint_3d
            valid[:21] = vis
        if (side == "left"):
            joint_2d_aug[21:42, :] = joint_2d
            joint_3d_aug[21:42, :] = joint_3d
            valid[21:42] = vis
        return joint_2d_aug, joint_3d_aug, valid

    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # h36m specific attributes
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}". '
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(
                data_cfg['camera_param_file'])
        
        actions = data_cfg.get('actions', '_all_')
        self.actions = set(
            actions if isinstance(actions, (list, tuple)) else [actions])
        
        subjects = data_cfg.get('subjects', '_all_')
        self.subjects = set(
            subjects if isinstance(subjects, (list, tuple)) else [subjects])
        self.subjects = subjects
    @staticmethod
    def get_bbox(uv):
        x = min(uv[:, 0]) - 10
        y = min(uv[:, 1]) - 10

        x_max = min(max(uv[:, 0]) + 10, 256)
        y_max = min(max(uv[:, 1]) + 10, 256)

        return [
            float(max(0, x)), float(max(0, y)), float(x_max - x), float(y_max - y)
        ]

    def _get_subsampling_ratio(self):
        return 6

    @staticmethod
    def _cam2pixel(cam_coord, f, c):
        x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
        y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
        z = cam_coord[:, 2]
        img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
        return img_coord
    
    def _get_partial_annotations(self, name):
        dataset = get_dataset(name)
        data_info = {
            'imgnames': [],
            'joints_3d': [],
            'joints_2d': [],
            'scales': [],
            'centers': [],
        }
        count = 0
        sampling_ratio = self._get_subsampling_ratio()
        flag = True
        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = np.load(sample['label_file'])
            fx = sample['intrinsics']['fx']
            fy = sample['intrinsics']['fy']
            cx = sample['intrinsics']['ppx']
            cy = sample['intrinsics']['ppy']

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            joint_3d = label["joint_3d"][0]
            joint_2d = label["joint_2d"][0]
            # Just extract the end part of the filename, not including root
            file_loc = sample['color_file'].split(self.ann_file)[1]
            joint_vis = np.where(
                np.all(joint_3d == -1, axis=1), 0, 1
            )
            subj, action, camera = DexYCBDataset._parse_dex_imgname(file_loc)
            joint_2d, joint_3d, joint_vis = self.left_or_right(joint_2d, joint_3d, sample["mano_side"], joint_vis)
            
            data_info["imgnames"].append(file_loc)
            joint_vis = np.expand_dims(joint_vis, axis=1)
            data_info["joints_3d"].append(np.hstack((joint_3d, joint_vis)))
            data_info["joints_2d"].append(np.hstack((joint_2d[:, :2], joint_vis)))
            bbox = DexYCBDataset.get_bbox(joint_2d)
            data_info["scales"].append(max(bbox[2], bbox[3]))
            center = [bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0]
            if (sample["mano_side"] == "right"):
                data_info["centers"].append(joint_2d[0, :2])
            else:
                data_info["centers"].append(joint_2d[21, :2])
            if (idx % 10 == 0):
                print(f"Idx: {idx}, len: {len(dataset)}, name: {name}")
        return data_info
    def load_annotations(self):
        """
            Reads AIST annotations, returns them in the following
            format: 

            data_info = {
                'imgnames': _imgnames,
                'joints_3d': _joints_3d,
                'joints_2d': _joints_2d,
                'scales': _scales,
                'centers': _centers,
            }
        """
        # get 2D joints
        if (self.test_mode):
            names = ['s0_test']
        else:
            names = ['s0_train', 's1_train', 's2_train', 's3_train']
        
        data_info = {
            'imgnames': [],
            'joints_3d': [],
            'joints_2d': [],
            'scales': [],
            'centers': [],
        }
        for name in names:
            temp_data_info = self._get_partial_annotations(name)
            data_info['imgnames'] += temp_data_info['imgnames']
            data_info['joints_3d'] += temp_data_info['joints_3d']
            data_info['joints_2d'] += temp_data_info['joints_2d']
            data_info['scales'] += temp_data_info['scales']
            data_info['centers'] += temp_data_info['centers']
            
        
        
        data_info["joints_3d"] = np.array(data_info["joints_3d"]).astype(np.float32)
        data_info["joints_2d"] = np.array(data_info["joints_2d"]).astype(np.float32)
        data_info["scales"] = np.array(data_info["scales"]).astype(np.float32)
        data_info["centers"] = np.array(data_info["centers"]).astype(np.float32)
        data_info["imgnames"] = np.array(data_info["imgnames"])
        return data_info

    @staticmethod
    def _parse_dex_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        Name format: 
        <subject_id>/<seq_id>/<camera>/color_<framenum>
        """
        splits = imgname.split("/")
        subj = splits[0]
        camera = splits[2]
        seq_id = splits[1]

        return subj, seq_id, camera

    def build_sample_indices(self):
        """Split original videos into sequences and build frame indices.

        This method overrides the default one in the base class.
        """

        # Group frames into videos. Assume that self.data_info is
        # chronological.
        video_frames = defaultdict(list)
        print(f"DEX len: {len(self.data_info['imgnames'])}")
        for idx, imgname in enumerate(self.data_info['imgnames']):
            subj, action, camera = DexYCBDataset._parse_dex_imgname(imgname)

            # TODO: Is _all_ going to be in self.actions and self.subjects?
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

    def _load_joint_2d_detection(self, det_file):
        """"Load 2D joint detection results from file."""
        joints_2d = np.load(det_file).astype(np.float32)

        return joints_2d

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
            np.set_printoptions(suppress=True)
            masks.append(np.ones((42, 1)))
            action = self._parse_dex_imgname(
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
                                    masks[indices], alignment)
            name_value_tuples.append((f'{err_name}_{action_category}', _error))

        return name_value_tuples
    
    def _load_K(self, x):
        return np.array(
            [[x['fx'], 0.0, x['ppx']], [0.0, x['fy'], x['ppy']], [0.0, 0.0, 1.0]]
        )
            
    def _load_camera_param(self, camera_param_file):
        subject_names = ["20200709-subject-01", "20200813-subject-02", "20200820-subject-03", 
                        "20200903-subject-04", "20200908-subject-05", "20200918-subject-06",
                        "20200928-subject-07", "20201002-subject-08", "20201015-subject-09",
                        "20201022-subject-10"]
        color_prefix = "color_"
        depth_prefix = "aligned_depth_to_color_"
        label_prefix = "labels_"
        camera_params = {}
        for subj in subject_names:
            all_names = glob.glob(f"{self._dex_ycb_dir}/{subj}/*/")
            for name in all_names:
                video_name = name.split("/")[-2]
                meta_file = self._dex_ycb_dir + '/' + subj + '/' + video_name + "/meta.yml"
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                serials = meta['serials']
                h = 480
                w = 640
                num_cameras = len(serials)
                data_dir = [
                    self._dex_ycb_dir + '/' + subj + '/' + video_name + '/' + s for s in serials
                ]

                num_frames = meta['num_frames']
                ycb_ids = meta['ycb_ids']
                mano_sides = meta['mano_sides']

                K = {}
                for s in serials:
                    intr_file = self._dex_ycb_dir + "/calibration/intrinsics/" + s + '_' + str(w) + 'x' + str(h) + ".yml"
                    with open(intr_file, 'r') as f:
                        intr = yaml.load(f, Loader=yaml.FullLoader)
                        K_single = self._load_K(intr['color'])
                        K[s] = K_single


                # Load extrinsics.
                extr_file = self._dex_ycb_dir + "/calibration/extrinsics_" + meta[
                    'extrinsics'] + "/extrinsics.yml"
                with open(extr_file, 'r') as f:
                      extr = yaml.load(f, Loader=yaml.FullLoader)
                T = extr['extrinsics']
                T = {
                    s: np.array(T[s], dtype=np.float32).reshape((3, 4)) for s in T
                }
                for s in serials:
                    R = T[s][:, :3]
                    t = T[s][:, 3]
                    K_single = K[s]
                    camera_params[(subj, video_name, s)] = {
                        "R": R,
                        "T": T,
                        "c": np.array([K_single[0][2], K_single[1][2]]),
                        "f": np.array([K_single[0][0], K_single[0][1]]),
                        'w': w,
                        'h': h
                    }
                    
                    
                    
                
        return camera_params
    
    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        subj, action, camera = self._parse_dex_imgname(imgname)
        return self.camera_param[(subj, action, camera)]
