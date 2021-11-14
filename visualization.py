import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def visualize_conf_matr(array):
    axis_labels = ['Not spam', 'Spam']
    df_cm = pd.DataFrame(array, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm,cmap="Blues", annot=True, annot_kws={"size": 12},fmt = 'g',xticklabels=axis_labels, yticklabels=axis_labels) # font size
    plt.show()
    return