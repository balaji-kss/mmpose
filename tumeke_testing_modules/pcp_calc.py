
'''
PCP - Percentage of Correct Parts¶
https://github.com/cbsudux/Human-Pose-Estimation-101#percentage-of-correct-parts---pcp

Percentage of Correct Parts - PCP
- A limb is considered detected and a correct part if the distance between the two predicted joint locations and the 
   true limb joint locations is at most half of the limb length (PCP at 0.5 )

Measures detection rate of limbs

Cons - penalizes shorter limbs

Calculation:
- For a specific part, PCP = (No. of correct parts for entire dataset) / (No. of total parts for entire dataset)
- Take a dataset with 10 images and 1 pose per image. Each pose has 8 parts - ( upper arm, lower arm, upper leg, lower leg ) x2
- No of upper arms = 10 * 2 = 20
- No of lower arms = 20
- No of lower legs = No of upper legs = 20
- If upper arm is detected correct for 17 out of the 20 upper arms i.e 17 ( 10 right arms and 7 left) → PCP = 17/20 = 85%

Higher the better


# 1) ground truth limb length

# 2) Euclidean distance between joints (already done) + avg between start joint and end joint euclidean distances

# 3) Check if limb ed is less than threshold (ground truth limb length * .5)
'''

'''Ground Truth Limb Length'''

import numpy as np
import pandas as pd
import seaborn as sns
from tumeke_testing_modules.load_gt_data import joint_order
from tumeke_testing_modules.load_model_data import euclidean_dist
from tumeke_testing_modules.pck_calc import get_ed_by_frame, frmt_perc

# def get_limb_diameter(r):
#     return euclidean_dist(np.array([r[x1], r[y1]]), np.array([r[x2], r[y2]]))


def get_pcp_stats(gt_df, df, model_name):
    ''''For a given data frame, compare with ground truth using PCK'''
    
    print('-' * 10)
    print(model_name)
    print('-' * 10)
    
    # Get ground truth limb length
    limbs = [
        {'name':'L_upper_arm', 'joints': [joint_order.index('L_Shoulder'), joint_order.index('L_Elbow')]},
        {'name':'R_upper_arm', 'joints': [joint_order.index('R_Shoulder'), joint_order.index('R_Elbow')]},
        {'name':'L_lower_arm', 'joints': [joint_order.index('L_Elbow'), joint_order.index('L_Wrist')]},
        {'name':'R_lower_arm', 'joints': [joint_order.index('R_Elbow'), joint_order.index('R_Wrist')]},
        {'name':'L_upper_leg', 'joints': [joint_order.index('L_Hip'), joint_order.index('L_Knee')]},
        {'name':'R_upper_leg', 'joints': [joint_order.index('R_Hip'), joint_order.index('R_Knee')]},
        {'name':'L_lower_leg', 'joints': [joint_order.index('L_Knee'), joint_order.index('L_Ankle')]},
        {'name':'R_lower_leg', 'joints': [joint_order.index('R_Knee'), joint_order.index('R_Ankle')]}
    ]
    pcp_gt_limb_length_df = gt_df[[]] # Return just index
    for limb in limbs:
        j1 = limb['joints'][0] # joint 1 index
        j2 = limb['joints'][1] # joint 2 index
        x1 = "j{}_x".format(j1)
        y1 = "j{}_y".format(j1)
        x2 = "j{}_x".format(j2)
        y2 = "j{}_y".format(j2)
        pcp_gt_limb_length_df[limb['name']] = \
            gt_df.apply(lambda r: euclidean_dist(np.array([r[x1], r[y1]]), np.array([r[x2], r[y2]])), axis=1)

    # PCP Theshold
    pcp_threshold_df = pcp_gt_limb_length_df * 0.50

    # Get Limb Euclidean Dist
    ed_dist_df = get_ed_by_frame(gt_df, df)
    limbs = [
        {'name':'L_upper_arm', 'joints': ['L_Shoulder', 'L_Elbow']},
        {'name':'R_upper_arm', 'joints': ['R_Shoulder', 'R_Elbow']},
        {'name':'L_lower_arm', 'joints': ['L_Elbow', 'L_Wrist']},
        {'name':'R_lower_arm', 'joints': ['R_Elbow', 'R_Wrist']},
        {'name':'L_upper_leg', 'joints': ['L_Hip', 'L_Knee']},
        {'name':'R_upper_leg', 'joints': ['R_Hip', 'R_Knee']},
        {'name':'L_lower_leg', 'joints': ['L_Knee', 'L_Ankle']},
        {'name':'R_lower_leg', 'joints': ['R_Knee', 'R_Ankle']}
    ]
    pcp_limb_ed_df = gt_df[[]] # Return just index
    for limb in limbs:
        j1 = limb['joints'][0] # joint 1 name
        j2 = limb['joints'][1] # joint 2 name

        pcp_limb_ed_df[limb['name']] = ed_dist_df['{}_ed'.format(j1)] + ed_dist_df['{}_ed'.format(j2)] / 2
        
        
    # Determine if body part is meeting pcp threshold
    pcp_df_scores = pcp_limb_ed_df.lt(pcp_threshold_df)
    pcp_df_scores = pcp_df_scores.applymap(lambda x: 1 if x == True else 0)

    # % Detected frames
    print('{} | % Detected frames:'.format(model_name),
        frmt_perc(pcp_df_scores.sum(axis=1).apply(lambda x: 1 if x > 0 else 0).sum() / pcp_df_scores.sum(axis=1).count()))

    # Total (all frames)
    print('{} | PCP - Total (all frames):'.format(model_name),
        frmt_perc(pcp_df_scores.sum(axis=1).sum() / pcp_df_scores.count(axis=1).sum()))

    # Total (detected frames)
    pcp_detected_df = pcp_df_scores.sum(axis=1).apply(lambda x: True if x > 0 else False)
    print('{} | PCP - Total (detected frames):'.format(model_name),
        frmt_perc(pcp_df_scores[pcp_detected_df].sum(axis=1).sum() / pcp_df_scores[pcp_detected_df].count(axis=1).sum()))

    # By Limb
    print('{} | PCP - By Limb (all frames):'.format(model_name))
    print(pcp_df_scores.sum() / pcp_df_scores.count() * 100)
    
    print('{} | PCP - By Limb (detected frames):'.format(model_name))
    print(pcp_df_scores[pcp_detected_df].sum() / pcp_df_scores[pcp_detected_df].count() * 100)

    # by Frame
    print('{} | PCP - By Frame:'.format(model_name))
    pd.DataFrame(pcp_df_scores).plot.area(title='{} | PCP over time by limb'.format(model_name))


def pcp_calc(gt_df, hrnet_df, wrnch_df):
    '''Calculate PCK stats for hrnet and wrnch'''
    
    print(
        '''
PCP - Percentage of Correct Parts¶
https://github.com/cbsudux/Human-Pose-Estimation-101#percentage-of-correct-parts---pcp

Percentage of Correct Parts - PCP
- A limb is considered detected and a correct part if the distance between the two predicted joint locations and the 
   true limb joint locations is at most half of the limb length (PCP at 0.5 )

Measures detection rate of limbs

Cons - penalizes shorter limbs

Calculation:
- For a specific part, PCP = (No. of correct parts for entire dataset) / (No. of total parts for entire dataset)
- Take a dataset with 10 images and 1 pose per image. Each pose has 8 parts - ( upper arm, lower arm, upper leg, lower leg ) x2
- No of upper arms = 10 * 2 = 20
- No of lower arms = 20
- No of lower legs = No of upper legs = 20
- If upper arm is detected correct for 17 out of the 20 upper arms i.e 17 ( 10 right arms and 7 left) → PCP = 17/20 = 85%

Higher the better
        '''
    )
    
    sns.set(rc={'figure.figsize':(21,7)})
    
    get_pcp_stats(gt_df, hrnet_df, 'hrnet')
    print('-' * 20)
    get_pcp_stats(gt_df, wrnch_df, 'wrnch')





