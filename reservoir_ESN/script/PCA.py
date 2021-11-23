import pandas as pd
import os
import random
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pandas import plotting 
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
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
    task_name = "approx3_1.0"
    df = df[df[task_name] < 1.02]
    f_exp = lambda x: math.exp(x)
    #df = df.query('function_name == "sinc"')
    df['L_log'].apply(f_exp)
    df['NL_log'].apply(f_exp)
    #[9,10,14,18,19,22,26, 36, 44, 54,57, 64, 72]
    #[26:44, 57:72]
    range1 = list(range(26, 46)) + list(range(54, 74))
    range1.extend([9, 10, 14, 18, 19, 22])
    dfs = df.iloc[:, range1].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    dfs.head()
    print(dfs)
    pca = PCA()
    pca.fit(dfs)
    # データを主成分空間に写像
    feature = pca.transform(dfs)
    pd.DataFrame(feature[:,:6], columns=["PC{}".format(x + 1) for x in range(6)]).head()
    print(feature)
    plotting.scatter_matrix(pd.DataFrame(feature[:,:6], 
                            columns=["PC{}".format(x + 1) for x in range(6)]), 
                            figsize=(8, 8), c=list(df.loc[:, task_name]), alpha=0.5) 
    plt.show()
    #寄与率
    pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    import matplotlib.ticker as ticker
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()
    # PCA の固有値
    pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    # PCA の固有ベクトル
    pd.DataFrame(pca.components_, columns=df.columns[range1], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    plt.figure(figsize=(6, 6))
    for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[range1]):
        plt.text(x, y, name)
    plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    #print(df.iloc[:, 258])
    #plotting.scatter_matrix(df.iloc[:, [9,10,11,12,13,14,18,19,20,21,22,23]], figsize=(8, 8), c=list(df.iloc[:, 258]), alpha=0.5, vmin = 0.0, vmax=1.0)
    #plt.show()
    break
    for function_name in function_names:
        df_sub = df.query('function_name == @function_name')
        df_sub = df_sub[df_sub[task_name] < 1.1]
        for i in range(20):
            for j in range(20):
                L_lower = i
                L_upper = (i + 1)
                NL_lower = j
                NL_upper = (j + 1)
                df_sub2 = df_sub.query('@NL_lower <= NLx_2 & NLx_2 < @NL_upper & @L_lower <= Lx_4 & Lx_4 < @L_upper')
                if len(df_sub2) > 10:
                    mean = df_sub2[task_name].mean()
                else:
                    mean = np.nan
                #print(i, j, mean, len(df_sub2))
                data[j, i] = mean
                if(len(df_sub2) > 10):
                    df_sub2.to_csv(function_name + str(i) + "_" + str(j) + ".csv")
        print(data)
        draw_heatmap(data, row_labels, column_labels, function_name)
    break
