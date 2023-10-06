import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tumeke_testing_modules.load_model_data import euclidean_dist, joint_order
plt.show()

def df_skeleton_smoothness(df, offset, model_name):
    ''' Get Euclidean distance between frame and previous frame'''
    
    # Build dict to accumulate calcs
    dict_2d_smth = {}
    for j in range(17):
            joint_field = 'j{}_euc_dist'.format(j)
            dict_2d_smth[joint_field] = []

    # Iterate through rows (Standard shift(-1) can't be used with this calc, so must iterate)
    for i in range(len(df)):
        for j in range(17):
            joint_field_x = 'j{}_x'.format(j)
            joint_field_y = 'j{}_y'.format(j)
            joint_field_score = 'j{}_score'.format(j)
            
            #TODO(znoland): return NaN when score below certain amount?
            
            try:
                dict_2d_smth['j{}_euc_dist'.format(j)].append(
                    euclidean_dist(
                        np.array([df.loc[i, joint_field_x], df.loc[i, joint_field_y]]), 
                        np.array([df.loc[i+offset, joint_field_x], df.loc[i+offset, joint_field_y]])
                    )
                )
            except:
                print('skiped row {}'.format(i))

    df_2d_smth = pd.DataFrame(dict_2d_smth)
    
    # Median distance between joints and previous frame by joint
    df_euc_by_joint = pd.DataFrame(df_2d_smth.median()).T
    df_euc_by_joint.columns = joint_order
    print('-' * 10)
    print('{} | Median distance between joints and previous frame by joint:'.format(model_name))
    print('-' * 10)
    print(df_euc_by_joint.T)
    
    return df_2d_smth
    

def vis_ed_by_joint(df_2d_smth, model_name):
    sns.set(rc={'figure.figsize':(21,15)})

    # df_2d_smth.j0_euc_dist.hist()
    # sns.histplot(df_2d_smth).show()

    n_rows=3
    n_cols=6
    # Create the subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.suptitle('{} | Euclidean distance between frames'.format(model_name))

    # Set x axis limit (keep all charts on same axis)
    for row in axes:
        for chart in row:
            chart.set_xlim(0,100) # df_2d_smth.quantile(0.90).max() # df_2d_smth.max().max() 

    for i, column in enumerate(df_2d_smth.columns):
        ax = sns.histplot(df_2d_smth[column],ax=axes[i//n_cols,i%n_cols])
        ax.set_title(joint_order[i]) 


def smoothness_calc(gt_df, hrnet_df, wrnch_df):
    '''Get smoothness for all data frames'''
    hrnet_ed_df = df_skeleton_smoothness(hrnet_df, offset=1, model_name='hrnet')
    wrnch_ed_df = df_skeleton_smoothness(wrnch_df, offset=1, model_name='wrnch')
    
    vis_ed_by_joint(hrnet_ed_df, model_name='hrnet')
    vis_ed_by_joint(wrnch_ed_df, model_name='wrnch')
    
    
    '''
       - filtered on the same frames as what is avaliable in the ground truth df
       - Because of this, offset 2 to get the every other frame (according to index)
    '''
#     df_skeleton_smoothness(gt_df, offset=2, model_name='ground truth')
#     df_skeleton_smoothness(hrnet_df.iloc[gt_df.index], offset=2, model_name='hrnet')
#     df_skeleton_smoothness(wrnch_df.iloc[gt_df.index], offset=2, model_name='wrnch')

          

          