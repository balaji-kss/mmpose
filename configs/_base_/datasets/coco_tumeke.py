joint_weights_body=[
        1., 1., 1., 1., 1., 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
        1.5
    ]
joint_weights_chin=[1.]
joint_weights_hand=[1.] * 24

sigmas_all=[
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
]

dataset_info = dict(
    dataset_name='coco_tumeke',
    paper_info=dict(
        author='Balaji Sundareshan',
        title='Tumeke',
        container='Tumeke',
        year='2023',
        homepage='https://www.tumeke.io/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        17:
        dict(name='face-8', id=17, color=[255, 255, 255], type='', swap=''),
        18:
        dict(
            name='left_forefinger1',
            id=18,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger1'),
        19:
        dict(
            name='left_forefinger2',
            id=19,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger2'),
        20:
        dict(
            name='left_forefinger3',
            id=20,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger3'),
        21:
        dict(
            name='left_forefinger4',
            id=21,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger4'),
        22:
        dict(
            name='left_middle_finger1',
            id=22,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger1'),
        23:
        dict(
            name='left_middle_finger2',
            id=23,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger2'),
        24:
        dict(
            name='left_middle_finger3',
            id=24,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger3'),
        25:
        dict(
            name='left_middle_finger4',
            id=25,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger4'),
        26:
        dict(
            name='left_ring_finger1',
            id=26,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger1'),
        27:
        dict(
            name='left_ring_finger2',
            id=27,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger2'),
        28:
        dict(
            name='left_ring_finger3',
            id=28,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger3'),
        29:
        dict(
            name='left_ring_finger4',
            id=29,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger4'),
        30:
        dict(
            name='right_forefinger1',
            id=30,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger1'),
        31:
        dict(
            name='right_forefinger2',
            id=31,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger2'),
        32:
        dict(
            name='right_forefinger3',
            id=32,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger3'),
        33:
        dict(
            name='right_forefinger4',
            id=33,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger4'),
        34:
        dict(
            name='right_middle_finger1',
            id=34,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger1'),
        35:
        dict(
            name='right_middle_finger2',
            id=35,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger2'),
        36:
        dict(
            name='right_middle_finger3',
            id=36,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger3'),
        37:
        dict(
            name='right_middle_finger4',
            id=37,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger4'),
        38:
        dict(
            name='right_ring_finger1',
            id=38,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger1'),
        39:
        dict(
            name='right_ring_finger2',
            id=39,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger2'),
        40:
        dict(
            name='right_ring_finger3',
            id=40,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger3'),
        41:
        dict(
            name='right_ring_finger4',
            id=41,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger4'),
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
        19:
        dict(
            link=('left_forefinger1', 'left_forefinger2'),
            id=19,
            color=[255, 153, 255]),
        20:
        dict(
            link=('left_forefinger2', 'left_forefinger3'),
            id=20,
            color=[255, 153, 255]),
        21:
        dict(
            link=('left_forefinger3', 'left_forefinger4'),
            id=21,
            color=[255, 153, 255]),
        22:
        dict(
            link=('left_middle_finger1', 'left_middle_finger2'),
            id=22,
            color=[102, 178, 255]),
        23:
        dict(
            link=('left_middle_finger2', 'left_middle_finger3'),
            id=23,
            color=[102, 178, 255]),
        24:
        dict(
            link=('left_middle_finger3', 'left_middle_finger4'),
            id=24,
            color=[102, 178, 255]),
        25:
        dict(
            link=('left_ring_finger1', 'left_ring_finger2'),
            id=25,
            color=[255, 51, 51]),
        26:
        dict(
            link=('left_ring_finger2', 'left_ring_finger3'),
            id=26,
            color=[255, 51, 51]),
        27:
        dict(
            link=('left_ring_finger3', 'left_ring_finger4'),
            id=27,
            color=[255, 51, 51]),
        28:
        dict(
            link=('right_forefinger1', 'right_forefinger2'),
            id=28,
            color=[255, 153, 255]),
        29:
        dict(
            link=('right_forefinger2', 'right_forefinger3'),
            id=29,
            color=[255, 153, 255]),
        30:
        dict(
            link=('right_forefinger3', 'right_forefinger4'),
            id=30,
            color=[255, 153, 255]),
        31:
        dict(
            link=('right_middle_finger1', 'right_middle_finger2'),
            id=31,
            color=[102, 178, 255]),
        32:
        dict(
            link=('right_middle_finger2', 'right_middle_finger3'),
            id=32,
            color=[102, 178, 255]),
        33:
        dict(
            link=('right_middle_finger3', 'right_middle_finger4'),
            id=33,
            color=[102, 178, 255]),
        34:
        dict(
            link=('right_ring_finger1', 'right_ring_finger2'),
            id=34,
            color=[255, 51, 51]),
        35:
        dict(
            link=('right_ring_finger2', 'right_ring_finger3'),
            id=35,
            color=[255, 51, 51]),
        36:
        dict(
            link=('right_ring_finger3', 'right_ring_finger4'),
            id=36,
            color=[255, 51, 51])
    },
    joint_weights=joint_weights_body + joint_weights_chin + joint_weights_hand,
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175',
    sigmas=sigmas_all[:17] + [sigmas_all[31]] + sigmas_all[96:108] + sigmas_all[117:129]
    )
