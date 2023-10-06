# ------------------------------------------------------------------------------
# Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import numpy as np


def nms(dets, thr):
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets: [[x1, y1, x2, y2, score]].
        thr: Retain overlap < thr.

    Returns:
         list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    joint_len = g.size
    if sigmas is None:
        if (joint_len == 17 * 3):
            sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                .87, .87, .89, .89
            ]) / 10.0
        else:
            sigmas = np.array([
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
                0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068, 0.066, 0.066,
                0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031,
                0.025, 0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045,
                0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
                0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017,
                0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010,
                0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009,
                0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
                0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
                0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019,
                0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
                0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
                0.019, 0.022, 0.031
            ]) / 10.0

    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def _rescore(overlap, scores, thr, type='gaussian'):
    """Rescoring mechanism gaussian or linear.

    Args:
        overlap: calculated ious
        scores: target scores.
        thr: retain oks overlap < thr.
        type: 'gaussian' or 'linear'

    Returns:
        np.ndarray: indexes to keep
    """
    assert len(overlap) == len(scores)
    assert type in ['gaussian', 'linear']

    if type == 'linear':
        inds = np.where(overlap >= thr)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(-overlap**2 / thr)

    return scores


def soft_oks_nms(kpts_db,
                 thr,
                 max_dets=20,
                 sigmas=None,
                 vis_thr=None,
                 score_per_joint=False):
    """Soft OKS NMS implementations.

    Args:
        kpts_db
        thr: retain oks overlap < thr.
        max_dets: max number of detections to keep.
        sigmas: Keypoint labelling uncertainty.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while len(order) > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        order = order[1:]
        scores = _rescore(oks_ovr, scores[1:], thr)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep
