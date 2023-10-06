import matplotlib.pyplot as plt
import seaborn as sns
from tumeke_testing_modules.load_model_data import joint_order
plt.show()



def visualize_df_scores(df):
    '''Visualize each joint's score in a given dataframe'''
    sns.set(rc={'figure.figsize':(21,15)})
    
    score_column_names = ['bbox_score']
    for i in range(17):
        score_column_names.append('j{}_score'.format(i))

    n_rows=3
    n_cols=6
    # Create the subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)

    # Set x axis limit (keep all charts on same axis)
    for row in axes:
        for chart in row:
            chart.set_xlim(0,1) 

    for i, column in enumerate(df[score_column_names].columns):
        ax = sns.histplot(df[column],ax=axes[i//n_cols,i%n_cols])
        if column == 'bbox_score':
            ax.set_title('bbox_score')
        else: 
            ax.set_title(joint_order[i-1]) # Currently has bbox as well, so need to offset title
            

 