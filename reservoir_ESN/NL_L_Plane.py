import pandas as pd
import os
from matplotlib import pyplot as plt
folder = "output_data2/"
for name in os.listdir(folder):
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.csv', sep=',',comment='#')
    #function_names = ["oddsinc"]
    function_names = ["sinc", "tanh"]
    task_name = "approx5_1.0"
    #df_sub = df.query('function_name == "sinc"')
    
    #print(temp)    
    statistics = df.describe()
    statistics.to_csv("statictics.csv")
    #df.to_csv('to_csv_out.csv')
    for function_name in function_names:
        df_sub = df.query('function_name == @function_name')
        data_x = df_sub.L
        data_y = df_sub.NL
        value = df_sub[task_name]
        print(value)
        plt.scatter(data_x, data_y,s=3, marker="o", c = value, alpha=0.3, linewidths=1, label = function_name, vmin=min(value), vmax=0.5)
    for function_name in function_names:
        df_sub = df.query('function_name == @function_name')
        temp = df_sub.apply(lambda x: list(x[x == x.min()].index))  
        min_nmse = df_sub.loc[temp[task_name][0]][task_name]
        min_nmse_L = df_sub.loc[temp[task_name][0]].L
        min_nmse_NL = df_sub.loc[temp[task_name][0]].NL
        if(function_name == "tanh"):
            marker = "x"
        if(function_name == "sinc"):
            marker = "*"
        plt.scatter(min_nmse_L, min_nmse_NL,s=300, marker=marker, c = min_nmse, alpha=1.0, linewidths=2, label = task_name + "_" + function_name + '{:.3f}'.format(min_nmse), vmax=0.5)
        print(min_nmse)
    plt.ylim(0, 400)
    plt.xlim(0, 45)
    plt.legend(loc = "best")
    plt.colorbar()

    #plt.show()
    plt.savefig(name + ".png", dpi = 600)
    plt.cla()

