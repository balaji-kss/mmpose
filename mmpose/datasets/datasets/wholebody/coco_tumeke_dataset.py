# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Optional

import numpy as np

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class CocoTumekeDataset(BaseCocoStyleDataset):
    """CocoTumeke dataset for pose estimation.

    COCO-Tumeke keypoints::

        0-16: 17 body keypoints,
        17: chin keypoint,
        18-41: 42 hand keypoints

        In total, we have 42 keypoints for tumeke pose estimation.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(
        from_file='configs/_base_/datasets/coco_tumeke.py')

    def get_tumeke_keypoints(self, keypoints):

        # body points + chin + left fingers + right fingers = 42 points
        kpts_body = keypoints[:, :17]
        kpts_chin = keypoints[:, 31].reshape((-1, 1, 3))
        kpts_lefthand = keypoints[:, 96:108]   
        kpts_righthand = keypoints[:, 117:129]

        keypoints = np.concatenate((kpts_body, kpts_chin, kpts_lefthand, kpts_righthand), axis=1)

        return keypoints

    def rm_out_kpts(self, keypoints, imh, imw):

        boolx = keypoints[0, :, 0] >= imw
        booly = keypoints[0, :, 1] >= imh

        keypoints[0, boolx | booly] = 0.0

        return keypoints

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        img_path = osp.join(self.data_prefix['img'], img['file_name'])
        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        # COCO-Wholebody: consisting of body, foot, face and hand keypoints
        _keypoints = np.array(ann['keypoints'] + ann['foot_kpts'] +
                              ann['face_kpts'] + ann['lefthand_kpts'] +
                              ann['righthand_kpts'], dtype=np.float32).reshape(1, -1, 3)

        _keypoints = self.get_tumeke_keypoints(_keypoints)
        _keypoints = self.rm_out_kpts(_keypoints, img_h, img_w)

        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        # num_keypoints = ann['num_keypoints']
        num_keypoints = int(np.sum(keypoints_visible == 1))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img_path,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann['iscrowd'],
            'segmentation': ann['segmentation'],
            'id': ann['id'],
            'category_id': ann['category_id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        return data_info
