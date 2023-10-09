# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import traceback
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmpose.datasets import DatasetInfo
import cv2
from mmpose.datasets.pipelines import Compose
from .inference import LoadImage, _box2cs, _xywh2xyxy, _xyxy2xywh
from mmpose.core.visualization import image
import pickle 
from mmpose.core import imshow_bboxes, imshow_keypoints, imshow_keypoints_3d
import mmcv
from mmcv.utils.misc import deprecated_api_warning
from mmpose.core.bbox import bbox_xywh2cs, bbox_xywh2xyxy, bbox_xyxy2xywh
from mmpose.datasets.pipelines import Compose

def write_2d_skel(pose2d_seq, output_num, tid, image_size):

    # Create a figure and an axes
    fig, ax = plt.subplots()
    print('Writing seq: ',  output_num)
    output = f"test_skel_{output_num}_{tid}.mp4"
    output_dir = f"test_skel_{output_num}_{tid}"

    connections = [
        (0, 1),
        (1, 4),
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 6),
        (1, 14),
        (4, 11),
        (11, 12),
        (12, 13),
        (14, 15),
        (15, 16),
        (8, 9),
        (9, 10),
        (14, 7),
        (7, 11),
        (14, 8),
        (8, 11),
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate each frame and save as an image file
    for t, joints in enumerate(pose2d_seq):
        plt.figure()
        for connection in connections:
            if pose2d_seq[t, connection[0], 2] <= 0 or pose2d_seq[t, connection[1], 2] <= 0:continue
            xs = [pose2d_seq[t, connection[0], 0], pose2d_seq[t, connection[1], 0]]
            ys = [pose2d_seq[t, connection[0], 1], pose2d_seq[t, connection[1], 1]]
            plt.plot(xs, ys, 'o-', color='green')
        plt.xlim(0, image_size[0])
        plt.ylim(image_size[1], 0)  # Invert Y axis to have the origin at the top-left corner
        plt.grid(True)
        plt.savefig(f'{output_dir}/frame{t}.jpg')
        plt.close()

    # Compile the frames into a video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video = cv2.VideoWriter(output, fourcc, 5, (640, 480))  # 5 is the FPS, (560, 400) is the frame size

    for i in range(len(pose2d_seq)):
        img = cv2.imread(f'{output_dir}/frame{i}.jpg')
        if img is None:continue
        video.write(img)

    video.release()
    cv2.destroyAllWindows()

    if os.path.exists(output_dir):
        # Remove the directory and all its contents
        shutil.rmtree(output_dir)
        print(f"The directory {output_dir} and all its contents have been deleted.")
    else:
        print(f"The directory {output_dir} does not exist.")
        
def extract_pose_sequence(pose_results, frame_idx, causal, seq_len, step=1):
    """Extract the target frame from 2D pose results, and pad the sequence to a
    fixed length.

    Args:
        pose_results (list[list[dict]]): Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required \
                    when ``with_track_id==True``.
                - bbox ((4, ) or (5, )): left, right, top, bottom, [score]

        frame_idx (int): The index of the frame in the original video.
        causal (bool): If True, the target frame is the last frame in
            a sequence. Otherwise, the target frame is in the middle of
            a sequence.
        seq_len (int): The number of frames in the input sequence.
        step (int): Step size to extract frames from the video.

    Returns:
        list[list[dict]]: Multi-frame pose detection results stored \
            in a nested list with a length of seq_len.
    """

    if causal:
        frames_left = seq_len - 1
        frames_right = 0
    else:
        frames_left = (seq_len - 1) // 2
        frames_right = frames_left
    num_frames = len(pose_results)

    # get the padded sequence
    pad_left = max(0, frames_left - frame_idx // step)
    pad_right = max(0, frames_right - (num_frames - 1 - frame_idx) // step)
    start = max(frame_idx % step, frame_idx - frames_left * step)
    end = min(num_frames - (num_frames - 1 - frame_idx) % step,
              frame_idx + frames_right * step + 1)
    pose_results_seq = [pose_results[0]] * pad_left + \
        pose_results[start:end:step] + [pose_results[-1]] * pad_right
    return pose_results_seq

def kpts2d_conf_mask(kpts_2d, conf_thresh):

    kpts_conf = kpts_2d[:, 2]
    kpts_conf = kpts_conf > conf_thresh
    kpts_conf = kpts_conf.astype('int')
    kpts_conf = np.concatenate((kpts_2d[:, :2], kpts_conf.reshape((17, 1))), axis=1)
    
    return kpts_conf

def _gather_pose_lifter_inputs(pose_results,
                               bbox_center,
                               bbox_scale,
                               norm_pose_2d=False,
                               conf_thresh = 0.35):
    """Gather input data (keypoints and track_id) for pose lifter model.

    Note:
        - The temporal length of the pose detection results: T
        - The number of the person instances: N
        - The number of the keypoints: K
        - The channel number of each keypoint: C

    Args:
        pose_results (List[List[Dict]]): Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required when
                    ``with_track_id==True```
                - bbox ((4, ) or (5, )): left, right, top, bottom, [score]

        bbox_center (ndarray[1, 2], optional): x, y. The average center
            coordinate of the bboxes in the dataset. `bbox_center` will be
            used only when `norm_pose_2d` is `True`.
        bbox_scale (int|float, optional): The average scale of the bboxes
            in the dataset.
            `bbox_scale` will be used only when `norm_pose_2d` is `True`.
        norm_pose_2d (bool): If True, scale the bbox (along with the 2D
            pose) to bbox_scale, and move the bbox (along with the 2D pose) to
            bbox_center. Default: False.

    Returns:
        list[list[dict]]: Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required when
                    ``with_track_id==True``
    """
    sequence_inputs = []
    for frame in pose_results:
        frame_inputs = []
        for res in frame:
            inputs = dict()

            if norm_pose_2d:
                bbox = res['bbox']
                center = np.array([[(bbox[0] + bbox[2]) / 2,
                                    (bbox[1] + bbox[3]) / 2]])
                scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                inputs['keypoints'] = (res['keypoints'][:, :2] - center) \
                    / scale * bbox_scale + bbox_center
            else:
                inputs['keypoints'] = res['keypoints'][:, :2]

            if res['keypoints'].shape[1] == 3:
                inputs['keypoints'] = np.concatenate(
                    [inputs['keypoints'], res['keypoints'][:, 2:]], axis=1)

            inputs['keypoints'] = kpts2d_conf_mask(inputs['keypoints'], conf_thresh)

            if 'track_id' in res:
                inputs['track_id'] = res['track_id']
            frame_inputs.append(inputs)
        sequence_inputs.append(frame_inputs)
    return sequence_inputs

def interpolate_missing_values(data, W):
    
    zero_indices = np.where(data[:, 0] == 0)[0]  # Assuming if one dimension is 0, all are 0
    
    if len(zero_indices) == 0:
        return data
    
    groups = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
    
    for group in groups:
        if len(group) > W:
            start = group[0]    
            data[start:] = data[start-1]
            break
        
        start, end = group[0], group[-1]
        
        # Handle edge cases where missing values are at the beginning or end of the array
        if start == 0:
            data[start:end+1] = data[end+1]
        elif end == len(data) - 1:
            data[start:end+1] = data[start-1]
        else:
            for i in range(data.shape[1]):
                data[start:end+1, i] = np.linspace(data[start-1, i], data[end+1, i], len(group)+2)[1:-1]
    
    return data

def fill_sequence(pose_results_2d, keypoints, tid, max_win_size):

    num_seq = len(pose_results_2d)
    T, K, C = keypoints.shape

    for fid in range(num_seq):
        for res in pose_results_2d[fid]:
            if res['track_id'] == tid:
                keypoints[fid] = res['keypoints']
    
    target_idx = int(num_seq // 2)

    # pad + interpolate sequence
    for k in range(K):
        # from target to left
        left_temp_seq = keypoints[:target_idx + 1, k, :]
        left_seq = left_temp_seq[::-1]
        interp_left_seq = interpolate_missing_values(left_seq, max_win_size)        
        keypoints[:target_idx + 1, k, :] = interp_left_seq[::-1]

        # from target to right
        right_temp_seq = keypoints[target_idx:, k, :]
        interp_right_seq = interpolate_missing_values(right_temp_seq, max_win_size)
        keypoints[target_idx:, k, :] = interp_right_seq

    return keypoints

def _collate_pose_sequence(pose_results, with_track_id=True, target_frame=-1, max_win_size=30):
    """Reorganize multi-frame pose detection results into individual pose
    sequences.

    Note:
        - The temporal length of the pose detection results: T
        - The number of the person instances: N
        - The number of the keypoints: K
        - The channel number of each keypoint: C

    Args:
        pose_results (List[List[Dict]]): Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required when
                    ``with_track_id==True```

        with_track_id (bool): If True, the element in pose_results is expected
            to contain "track_id", which will be used to gather the pose
            sequence of a person from multiple frames. Otherwise, the pose
            results in each frame are expected to have a consistent number and
            order of identities. Default is True.
        target_frame (int): The index of the target frame. Default: -1.
    """
    T = len(pose_results)
    assert T > 0

    target_frame = (T + target_frame) % T  # convert negative index to positive

    N = len(pose_results[target_frame])  # use identities in the target frame
    if N == 0:
        return []

    K, C = pose_results[target_frame][0]['keypoints'].shape #(17, 3)

    track_ids = None
    if with_track_id:
        track_ids = [res['track_id'] for res in pose_results[target_frame]]

    pose_sequences = []
    for idx in range(N):
        pose_seq = dict()
        # gather static information
        for k, v in pose_results[target_frame][idx].items():
            if k != 'keypoints':
                pose_seq[k] = v
        # gather keypoints
        if not with_track_id:
            pose_seq['keypoints'] = np.stack(
                [frame[idx]['keypoints'] for frame in pose_results])
        else:
            keypoints = np.zeros((T, K, C), dtype=np.float32)
            keypoints[target_frame] = pose_results[target_frame][idx][
                'keypoints']
            keypoints = fill_sequence(pose_results, keypoints, track_ids[idx], max_win_size)
            pose_seq['keypoints'] = keypoints
        pose_sequences.append(pose_seq)

    return pose_sequences

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


def inference_pose_lifter_model(model,
                                pose_results_2d,
                                dataset=None,
                                dataset_info=None,
                                with_track_id=True,
                                image_size=None,
                                norm_pose_2d=False,
                                conf_thresh = 0.35,
                                output_num=0,
                                trt=True):
    """Inference 3D pose from 2D pose sequences using a pose lifter model.

    Args:
        model (nn.Module): The loaded pose lifter model
        pose_results_2d (list[list[dict]]): The 2D pose sequences stored in a
            nested list. Each element of the outer list is the 2D pose results
            of a single frame, and each element of the inner list is the 2D
            pose of one person, which contains:

            - "keypoints" (ndarray[K, 2 or 3]): x, y, [score]
            - "track_id" (int)
        dataset (str): Dataset name, e.g. 'Body3DH36MDataset'
        with_track_id: If True, the element in pose_results_2d is expected to
            contain "track_id", which will be used to gather the pose sequence
            of a person from multiple frames. Otherwise, the pose results in
            each frame are expected to have a consistent number and order of
            identities. Default is True.
        image_size (tuple|list): image width, image height. If None, image size
            will not be contained in dict ``data``.
        norm_pose_2d (bool): If True, scale the bbox (along with the 2D
            pose) to the average bbox scale of the dataset, and move the bbox
            (along with the 2D pose) to the average bbox center of the dataset.

    Returns:
        list[dict]: 3D pose inference results. Each element is the result of \
            an instance, which contains:

            - "keypoints_3d" (ndarray[K, 3]): predicted 3D keypoints
            - "keypoints" (ndarray[K, 2 or 3]): from the last frame in \
                ``pose_results_2d``.
            - "track_id" (int): from the last frame in ``pose_results_2d``. \
                If there is no valid instance, an empty list will be \
                returned.
    """
    cfg = model.cfg
    test_pipeline = Compose(cfg.test_pipeline)

    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    if dataset_info is not None:
        dataset_info_obj = DatasetInfo(dataset_info)
        flip_pairs = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
        bbox_center = np.array([[528, 427]], dtype=np.float32)
        bbox_scale = 400
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        if dataset == 'Body3DH36MDataset' or dataset == 'Body3DH36MModifiedDataset' or dataset=='Body3DCombinedDataset':
            flip_pairs = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
            bbox_center = np.array([[528, 427]], dtype=np.float32)
            bbox_scale = 400
        else:
            raise NotImplementedError()
    if trt:
        target_idx = len(pose_results_2d) // 2
    else:
        target_idx = -1 if model.causal else len(pose_results_2d) // 2
    pose_lifter_inputs = _gather_pose_lifter_inputs(pose_results_2d,
                                                    bbox_center, bbox_scale,
                                                    norm_pose_2d, conf_thresh = conf_thresh)
    pose_sequences_2d = _collate_pose_sequence(pose_lifter_inputs,
                                               with_track_id, target_idx)

    if not pose_sequences_2d:
        return []

    batch_data = []
    count = 0
    for seq in pose_sequences_2d:
        pose_2d = seq['keypoints'].astype(np.float32)
        T, K, C = pose_2d.shape #(243, 17, 3)

        input_2d = pose_2d[:, :, :2]
        input_2d_visible = pose_2d[:, :, 2]

        target = np.zeros((K, 3), dtype=np.float32)
        target_visible = pose_2d[target_idx, :, 2].reshape((-1, 1)).astype(np.float32)
        target_image_path = None
        
        data = {
            'input_2d': input_2d,
            'input_2d_visible': input_2d_visible,
            'target': target,
            'target_visible': target_visible,
            'target_image_path': target_image_path,
            'ann_info': {
                'num_joints': K,
                'flip_pairs': flip_pairs
            }
        }

        if image_size is not None:
            assert len(image_size) == 2
            data['image_width'] = image_size[0]
            data['image_height'] = image_size[1]
        
        data = test_pipeline(data)
        batch_data.append(data)

        # test visualize input
        # tid = seq['track_id']
        # if output_num % 30 == 0 and tid == 1:
        #     write_2d_skel(pose_2d, output_num, tid = tid, image_size=image_size)
            
    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    if trt:
        device = "cuda:0"
        batch_data = scatter(batch_data, target_gpus=[device.index])[0]
    else:
        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
            batch_data = scatter(batch_data, target_gpus=[device.index])[0]
        else:
            batch_data = scatter(batch_data, target_gpus=[-1])[0]
    if trt:
        poses_3d = []
        for i in range(batch_data["input"].shape[0]):
            with torch.no_grad():
                trt_outputs = model({'input.1': batch_data["input"][i][None, :]})
            output = trt_outputs['116']
            poses_3d.append(output.cpu().detach().numpy()[0])
        poses_3d = np.array(poses_3d)
    else:
        with torch.no_grad():
            result = model(
                input=batch_data['input'],
                metas=batch_data['metas'],
                return_loss=False)

        poses_3d = result['preds']
    if poses_3d.shape[-1] != 4:
        assert poses_3d.shape[-1] == 3
        dummy_score = np.ones(
            poses_3d.shape[:-1] + (1, ), dtype=poses_3d.dtype)
        poses_3d = np.concatenate((poses_3d, dummy_score), axis=-1)
    pose_results = []
    for pose_2d, pose_3d in zip(pose_sequences_2d, poses_3d):
        pose_result = pose_2d.copy()
        pose_result['keypoints_3d'] = pose_3d
        pose_results.append(pose_result)

    return pose_results

def show_result(result,
                img=None,
                skeleton=None,
                pose_kpt_color=None,
                pose_link_color=None,
                radius=8,
                thickness=2,
                vis_height=400,
                num_instances=-1,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Visualize 3D pose estimation results.

    Args:
        result (list[dict]): The pose estimation results containing:
            - "keypoints_3d" ([K,4]): 3D keypoints
            - "keypoints" ([K,3] or [T,K,3]): Optional for visualizing
                2D inputs. If a sequence is given, only the last frame
                will be used for visualization
            - "bbox" ([4,] or [T,4]): Optional for visualizing 2D inputs
            - "title" (str): title for the subplot
        img (str or Tensor): Optional. The image to visualize 2D inputs on.
        skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
            links, each is a pair of joint indices.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links.
            If None, do not draw links.
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        vis_height (int): The image height of the visualization. The width
            will be N*vis_height depending on the number of visualized
            items.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        Tensor: Visualized img, only if not `show` or `out_file`.
    """
    if num_instances < 0:
        assert len(result) > 0
    result = sorted(result, key=lambda x: x.get('track_id', 1e4))

    # draw image and input 2d poses
    if img is not None:
        img = mmcv.imread(img)

        bbox_result = []
        pose_input_2d = []
        for res in result:
            if 'bbox' in res:
                bbox = np.array(res['bbox'])
                if bbox.ndim != 1:
                    assert bbox.ndim == 2
                    bbox = bbox[-1]  # Get bbox from the last frame
                bbox_result.append(bbox)
            if 'keypoints' in res:
                kpts = np.array(res['keypoints'])
                if kpts.ndim != 2:
                    assert kpts.ndim == 3
                    kpts = kpts[-1]  # Get 2D keypoints from the last frame
                pose_input_2d.append(kpts)

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            imshow_bboxes(
                img,
                bboxes,
                colors='green',
                thickness=thickness,
                show=False)
        if len(pose_input_2d) > 0:
            imshow_keypoints(
                img,
                pose_input_2d,
                skeleton,
                kpt_score_thr=0.3,
                pose_kpt_color=pose_kpt_color,
                pose_link_color=pose_link_color,
                radius=radius,
                thickness=thickness)
        img = mmcv.imrescale(img, scale=vis_height / img.shape[0])

    img_vis = imshow_keypoints_3d(
        result,
        img,
        skeleton,
        pose_kpt_color,
        pose_link_color,
        vis_height,
        num_instances=num_instances)

    if show:
        mmcv.visualization.imshow(img_vis, win_name, wait_time)

    if out_file is not None:
        mmcv.imwrite(img_vis, out_file)

    return img_vis

    
def vis_3d_pose_result(model,
                       result,
                       img=None,
                       dataset='Body3DH36MDataset',
                       dataset_info=None,
                       kpt_score_thr=0.3,
                       radius=8,
                       thickness=2,
                       vis_height=400,
                       num_instances=-1,
                       axis_azimuth=70,
                       show=False,
                       out_file=None):
    """Visualize the 3D pose estimation results.

    Args:
        model (nn.Module): The loaded model.
        result (list[dict])
    """
    if dataset_info is not None:
        skeleton = dataset_info.skeleton
        pose_kpt_color = dataset_info.pose_kpt_color
        pose_link_color = dataset_info.pose_link_color
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
        
        if dataset == 'Body3DCombinedDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                        [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                        [8, 14], [14, 15], [15, 16]]

            pose_kpt_color = palette[[
                9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
            ]]
            pose_link_color = palette[[
                0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
            ]]
            
#             skeleton = [[0, 1], [1, 3], [2, 4], [0, 5], [0, 6], [5, 6], [5, 7], [6, 8],
#                         [7, 9], [8, 10], [5, 11], [6, 12], [11, 13], [12, 14],
#                         [13, 15], [14, 16], [11, 17], [12, 17]]

#             pose_kpt_color = palette[[
#                 9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0, 0, 0
#             ]]
#             pose_link_color = palette[[
#                 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0, 0, 0
#             ]]
        elif dataset == 'Body3DH36MDataset' or dataset == 'Body3DH36MModifiedDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                        [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                        [8, 14], [14, 15], [15, 16]]

            pose_kpt_color = palette[[
                9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
            ]]
            pose_link_color = palette[[
                0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
            ]]
        
        elif dataset == 'DexYCBDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], 
            [0, 5], [5, 6], [6, 7], [7, 8], 
            [0, 9], [9, 10], [10, 11], [11, 12], 
            [0, 13], [13, 14], [14, 15], [0, 16], 
            [16, 17], [17, 18], [18, 19], [19, 20],
            [21, 22], [22, 23], [23, 24], [24, 25],
            [21, 26], [26, 27], [27, 28], [28, 29],
            [21, 30], [30, 31], [31, 32], [32, 33],
            [21, 34], [34, 35], [35, 36], [21, 37], [37, 38],
            [38, 39], [39, 40], [40, 41]]

            pose_kpt_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                              [14, 128, 250], [80, 127, 255], [80, 127, 255],
                              [80, 127, 255], [80, 127, 255], [71, 99, 255],
                              [71, 99, 255], [71, 99, 255], [71, 99, 255],
                              [0, 36, 255], [0, 36, 255], [0, 36, 255],
                              [0, 36, 255], [0, 0, 230], [0, 0, 230],
                              [0, 0, 230], [0, 0, 230], [0, 0, 139],
                              [237, 149, 100], [237, 149, 100],
                              [237, 149, 100], [237, 149, 100], [230, 128, 77],
                              [230, 128, 77], [230, 128, 77], [230, 128, 77],
                              [255, 144, 30], [255, 144, 30], [255, 144, 30],
                              [255, 144, 30], [153, 51, 0], [153, 51, 0],
                              [153, 51, 0], [153, 51, 0], [255, 51, 13],
                              [255, 51, 13], [255, 51, 13], [255, 51, 13],
                              [103, 37, 8]]

            pose_link_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                               [14, 128, 250], [80, 127, 255], [80, 127, 255],
                               [80, 127, 255], [80, 127, 255], [71, 99, 255],
                               [71, 99, 255], [71, 99, 255], [71, 99, 255],
                               [0, 36, 255], [0, 36, 255], [0, 36, 255],
                               [0, 36, 255], [0, 0, 230], [0, 0, 230],
                               [0, 0, 230], [0, 0, 230], [237, 149, 100],
                               [237, 149, 100], [237, 149, 100],
                               [237, 149, 100], [230, 128, 77], [230, 128, 77],
                               [230, 128, 77], [230, 128, 77], [255, 144, 30],
                               [255, 144, 30], [255, 144, 30], [255, 144, 30],
                               [153, 51, 0], [153, 51, 0], [153, 51, 0],
                               [153, 51, 0], [255, 51, 13], [255, 51, 13],
                               [255, 51, 13], [255, 51, 13]]
            
        elif dataset == 'InterHand3DDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 20], [4, 5], [5, 6],
                        [6, 7], [7, 20], [8, 9], [9, 10], [10, 11], [11, 20],
                        [12, 13], [13, 14], [14, 15], [15, 20], [16, 17],
                        [17, 18], [18, 19], [19, 20], [21, 22], [22, 23],
                        [23, 24], [24, 41], [25, 26], [26, 27], [27, 28],
                        [28, 41], [29, 30], [30, 31], [31, 32], [32, 41],
                        [33, 34], [34, 35], [35, 36], [36, 41], [37, 38],
                        [38, 39], [39, 40], [40, 41]]

            pose_kpt_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                              [14, 128, 250], [80, 127, 255], [80, 127, 255],
                              [80, 127, 255], [80, 127, 255], [71, 99, 255],
                              [71, 99, 255], [71, 99, 255], [71, 99, 255],
                              [0, 36, 255], [0, 36, 255], [0, 36, 255],
                              [0, 36, 255], [0, 0, 230], [0, 0, 230],
                              [0, 0, 230], [0, 0, 230], [0, 0, 139],
                              [237, 149, 100], [237, 149, 100],
                              [237, 149, 100], [237, 149, 100], [230, 128, 77],
                              [230, 128, 77], [230, 128, 77], [230, 128, 77],
                              [255, 144, 30], [255, 144, 30], [255, 144, 30],
                              [255, 144, 30], [153, 51, 0], [153, 51, 0],
                              [153, 51, 0], [153, 51, 0], [255, 51, 13],
                              [255, 51, 13], [255, 51, 13], [255, 51, 13],
                              [103, 37, 8]]

            pose_link_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                               [14, 128, 250], [80, 127, 255], [80, 127, 255],
                               [80, 127, 255], [80, 127, 255], [71, 99, 255],
                               [71, 99, 255], [71, 99, 255], [71, 99, 255],
                               [0, 36, 255], [0, 36, 255], [0, 36, 255],
                               [0, 36, 255], [0, 0, 230], [0, 0, 230],
                               [0, 0, 230], [0, 0, 230], [237, 149, 100],
                               [237, 149, 100], [237, 149, 100],
                               [237, 149, 100], [230, 128, 77], [230, 128, 77],
                               [230, 128, 77], [230, 128, 77], [255, 144, 30],
                               [255, 144, 30], [255, 144, 30], [255, 144, 30],
                               [153, 51, 0], [153, 51, 0], [153, 51, 0],
                               [153, 51, 0], [255, 51, 13], [255, 51, 13],
                               [255, 51, 13], [255, 51, 13]]
        else:
            raise NotImplementedError

    try:
        img = show_result(
            result,
            img,
            skeleton,
            radius=radius,
            thickness=thickness,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            num_instances=num_instances,
            show=show,
            out_file=out_file)
    except:
        print(traceback.format_exc(), flush=True)
        return img
    
    return img


def inference_interhand_3d_model(model,
                                 img_or_path,
                                 det_results,
                                 bbox_thr=None,
                                 format='xywh',
                                 dataset='InterHand3DDataset'):
    """Inference a single image with a list of hand bounding boxes.

    Note:
        - num_bboxes: N
        - num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        det_results (list[dict]): The 2D bbox sequences stored in a list.
            Each each element of the list is the bbox of one person, whose
            shape is (ndarray[4 or 5]), containing 4 box coordinates
            (and score).
        dataset (str): Dataset name.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        list[dict]: 3D pose inference results. Each element is the result \
            of an instance, which contains the predicted 3D keypoints with \
            shape (ndarray[K,3]). If there is no valid instance, an \
            empty list will be returned.
    """

    assert format in ['xyxy', 'xywh']

    pose_results = []

    if len(det_results) == 0:
        return pose_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = bbox_xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    if dataset == 'InterHand3DDataset':
        flip_pairs = [[i, 21 + i] for i in range(21)]
    else:
        raise NotImplementedError()

    batch_data = []
    for bbox in bboxes:
        image_size = cfg.data_cfg.image_size
        aspect_ratio = image_size[0] / image_size[1]  # w over h
        center, scale = bbox_xywh2cs(bbox, aspect_ratio, padding=1.25)

        # prepare data
        data = {
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs,
                'heatmap3d_depth_bound': cfg.data_cfg['heatmap3d_depth_bound'],
                'heatmap_size_root': cfg.data_cfg['heatmap_size_root'],
                'root_depth_bound': cfg.data_cfg['root_depth_bound']
            }
        }

        if isinstance(img_or_path, np.ndarray):
            data['img'] = img_or_path
        else:
            data['image_file'] = img_or_path

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False)

    poses_3d = result['preds']
    rel_root_depth = result['rel_root_depth']
    hand_type = result['hand_type']
    if poses_3d.shape[-1] != 4:
        assert poses_3d.shape[-1] == 3
        dummy_score = np.ones(
            poses_3d.shape[:-1] + (1, ), dtype=poses_3d.dtype)
        poses_3d = np.concatenate((poses_3d, dummy_score), axis=-1)

    # add relative root depth to left hand joints
    poses_3d[:, 21:, 2] += rel_root_depth

    # set joint scores according to hand type
    poses_3d[:, :21, 3] *= hand_type[:, [0]]
    poses_3d[:, 21:, 3] *= hand_type[:, [1]]

    pose_results = []
    for pose_3d, person_res, bbox_xyxy in zip(poses_3d, det_results,
                                              bboxes_xyxy):
        pose_res = person_res.copy()
        pose_res['keypoints_3d'] = pose_3d
        pose_res['bbox'] = bbox_xyxy
        pose_results.append(pose_res)

    return pose_results


def inference_mesh_model(model,
                         img_or_path,
                         det_results,
                         bbox_thr=None,
                         format='xywh',
                         dataset='MeshH36MDataset'):
    """Inference a single image with a list of bounding boxes.

    Note:
        - num_bboxes: N
        - num_keypoints: K
        - num_vertices: V
        - num_faces: F

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        det_results (list[dict]): The 2D bbox sequences stored in a list.
            Each element of the list is the bbox of one person.
            "bbox" (ndarray[4 or 5]): The person bounding box,
            which contains 4 box coordinates (and score).
        bbox_thr (float | None): Threshold for bounding boxes.
            Only bboxes with higher scores will be fed into the pose
            detector. If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

            - 'xyxy' means (left, top, right, bottom),
            - 'xywh' means (left, top, width, height).
        dataset (str): Dataset name.

    Returns:
        list[dict]: 3D pose inference results. Each element \
            is the result of an instance, which contains:

            - 'bbox' (ndarray[4]): instance bounding bbox
            - 'center' (ndarray[2]): bbox center
            - 'scale' (ndarray[2]): bbox scale
            - 'keypoints_3d' (ndarray[K,3]): predicted 3D keypoints
            - 'camera' (ndarray[3]): camera parameters
            - 'vertices' (ndarray[V, 3]): predicted 3D vertices
            - 'faces' (ndarray[F, 3]): mesh faces

            If there is no valid instance, an empty list
            will be returned.
    """

    assert format in ['xyxy', 'xywh']

    pose_results = []

    if len(det_results) == 0:
        return pose_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = bbox_xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    if dataset == 'MeshH36MDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                      [20, 21], [22, 23]]
    else:
        raise NotImplementedError()

    batch_data = []
    for bbox in bboxes_xywh:
        image_size = cfg.data_cfg.image_size
        aspect_ratio = image_size[0] / image_size[1]  # w over h
        center, scale = bbox_xywh2cs(bbox, aspect_ratio, padding=1.25)

        # prepare data
        data = {
            'image_file':
            img_or_path,
            'center':
            center,
            'scale':
            scale,
            'rotation':
            0,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'dataset':
            dataset,
            'joints_2d':
            np.zeros((cfg.data_cfg.num_joints, 2), dtype=np.float32),
            'joints_2d_visible':
            np.zeros((cfg.data_cfg.num_joints, 1), dtype=np.float32),
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'pose':
            np.zeros(72, dtype=np.float32),
            'beta':
            np.zeros(10, dtype=np.float32),
            'has_smpl':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs,
            }
        }

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, target_gpus=[device])[0]

    # forward the model
    with torch.no_grad():
        preds = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_vertices=True,
            return_faces=True)

    for idx in range(len(det_results)):
        pose_res = det_results[idx].copy()
        pose_res['bbox'] = bboxes_xyxy[idx]
        pose_res['center'] = batch_data['img_metas'][idx]['center']
        pose_res['scale'] = batch_data['img_metas'][idx]['scale']
        pose_res['keypoints_3d'] = preds['keypoints_3d'][idx]
        pose_res['camera'] = preds['camera'][idx]
        pose_res['vertices'] = preds['vertices'][idx]
        pose_res['faces'] = preds['faces']
        pose_results.append(pose_res)
    return pose_results


def vis_3d_mesh_result(model, result, img=None, show=False, out_file=None):
    """Visualize the 3D mesh estimation results.

    Args:
        model (nn.Module): The loaded model.
        result (list[dict]): 3D mesh estimation results.
    """
    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(result, img, show=show, out_file=out_file)

    return img
