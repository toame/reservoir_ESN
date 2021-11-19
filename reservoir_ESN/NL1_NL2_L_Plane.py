import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
#folder = "output_data/"
folder = "output_data20211030_2/"
fig = plt.figure()
ax = Axes3D(fig)

for name in os.listdir(folder):
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.csv', sep=',',comment='#')
    #df = df.query('seed == 0')
    #df = df.query('p == 1.0')
    #df = df.query('bias_factor > 0')
    #df = df.query('input_signal_factor >= 10.0')
    #df = df.query('p == 0.9')
    #df = df.query('NL2 > 0.95')
    #function_names = ["oddsinc"]
    function_names = ["sinc", "tanh"]
    task_name = "narma10"
    #df_sub = df.query('function_name == "sinc"')
    
    #print(temp)    
    statistics = df.describe()
    statistics.to_csv("statictics.csv")
    #df.to_csv('to_csv_out.csv')
    ax.set_xlabel("NL")
    ax.set_ylabel("NL_log")
    ax.set_zlabel("L")

    for function_name in function_names:
        df_sub = df.query('function_name == @function_name')
        data_x = df_sub.NL
        data_y = df_sub.NL_log
        data_z = df_sub.L
        value = df_sub.p
        print(value)
        #plt.scatter(data_x, data_y,s=3, marker="o", c = value, alpha=0.4, linewidths=1, label = function_name, vmin = 0.1, vmax = 1.0)
        #plt.scatter(data_x, data_y,s=3, marker="o", c = value, alpha=0.5, linewidths=1, label = function_name, norm=LogNorm(vmin=0.01, vmax=1.0), cmap='viridis_r')
        #plt.scatter(data_x, data_y,s=3, marker="o", alpha=0.3, linewidths=1, label = function_name)
        ax.plot(data_x,data_y,data_z, marker="o",linestyle='None')
    plt.show()
    plt.clf()
    break

