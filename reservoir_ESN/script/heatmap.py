import pandas as pd
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
folder = "../output_data/"

def draw_heatmap(data, row_labels, column_labels, name):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.jet, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    cmap = plt.cm.jet
    cmap.set_bad('white',1.)

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig('image_' + function_name + '.png', dpi = 200)

data = np.random.uniform(0, 1.0, (20, 20))
row_labels = range(0, 40, 2)
column_labels = range(0, 20, 1)

for name in os.listdir(folder):
    print(name)
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.csv', sep=',',comment='#')
    function_names = ["tanh", "sinc"]
    task_name = "approx6_0.0"
    
    for function_name in function_names:
        df_sub = df.query('function_name == @function_name')
        df_sub = df_sub[df_sub[task_name] < 1.1]
        for i in range(20):
            for j in range(20):
                L_lower = i * 2
                L_upper = (i + 1) * 2
                NL_lower = j * 1
                NL_upper = (j + 1) * 1
                df_sub2 = df_sub.query('@NL_lower <= NLx_3 & NLx_3 < @NL_upper & @L_lower <= Lx_4 & Lx_4 < @L_upper')
                if len(df_sub2) > 0:
                    mean = df_sub2[task_name].mean()
                else:
                    mean = np.nan
                #print(i, j, mean, len(df_sub2))
                data[j, i] = mean
        print(data)
        draw_heatmap(data, row_labels, column_labels, function_name)
    break
