from matplotlib.pyplot import figure
import matplotlib.image as mpimg 
from scipy import ndimage
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib.patches as mpatches

def get_size(df, df_name):
    frame_img_name = re.search(r'(\d+[.][jpg]+)', df['img'].values[0]).group(0)
    file_img_name = re.findall(r'/([0-9a-zA-Z-_]+)', df[0:1].img.values[0])[1]
    
    img = mpimg.imread('work_dirs/tumeke_testing/ground_truth_images/{}/{}'.format(file_img_name, frame_img_name))
    return img.shape

def visualize_overlap(gt_df, other_dfs, df_names, frame_number):
    '''Visualizes a given frame with the ground truth, hrnet, and wrnch estimates overlayed on top'''
    
    sns.set(rc={'figure.figsize':(21,15)})
    
    gt_f = gt_df.loc[[frame_number]]
    
    disp_frames = []
    for df in other_dfs:
        disp_frames.append(df.loc[[frame_number]])
    
    frame_img_name = re.search(r'(\d+[.][jpg]+)', gt_f['img'].values[0]).group(0)
    file_img_name = re.findall(r'/([0-9a-zA-Z-_]+)', gt_df[0:1].img.values[0])[1]
    img = mpimg.imread('work_dirs/tumeke_testing/ground_truth_images/{}/{}'.format(file_img_name, frame_img_name))
    colors = ["red", "green", "yellow"]
    for j in range(17):
        x = "j{}_x".format(j)
        y = "j{}_y".format(j)
        l = "j{}_l".format(j)
        # Ground Truth - BLUE
        p1 = sns.scatterplot(data=gt_f, x=x, y=y, color='blue')
        p1.text(gt_f[x]-70, gt_f[y], 
             gt_f[l].values[0], horizontalalignment='left', 
             size=7, color='blue', weight='regular')
        c = 0
        for df in disp_frames:
            p2 = sns.scatterplot(data=df, x=x, y=y, color=colors[c])
            p2.text(df[x]+20, df[y]-1, 
                 df[l].values[0], horizontalalignment='left', 
                 size=7, color=colors[c], weight='regular')
            c += 1

    patches = [mpatches.Patch(color='blue', label='Ground Truth')]
    c = 0
    for df in other_dfs:
        patches.append(mpatches.Patch(color=colors[c], label=df_names[c]))
        c += 1
    plt.legend(handles=patches)
        
    plt.imshow(img)
    plt.show()
    
