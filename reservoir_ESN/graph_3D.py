import pandas as pd
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


folder = "nmse_gain_data/"
for name in os.listdir(folder):
    #name = "approx_3_3.0_100"
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.txt', sep=',',comment='#')
    print(df)
    #p_list = df["p"].unique()
    #input_gain_list = df["input_gain"].unique()
    #feed_gain_list = df["feed_gain"].unique()
    #nmse_list = df["nmse"].unique()
    function_names = ["mackey200"];

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #for function_name in function_names:
    #data_x, data_y, data_z = [], [], []

 
    #array_a = np.genfromtxt( folder + name + '.txt' , skip_header = 1 , delimiter = ',' )

    xdata = df['input_gain']
    ydata = df['feed_gain']
    zdata = df['nmse']

      #  for input_gain in input_gain_list:
         #   for feed_gain in feed_gain_list:
            #    for nmse in nmse_list:
               #     data_x.append(input_gain)
                #    data_y.append(feed_gain)
                 #   data_z.append(nmse)
      #  plt.xlim(0, 10.0)
       # plt.ylim(0, 10.0)
        #plt.zscale("log")
       # plt.plot(data_x, data_y, marker="o", label = function_name)
        #ax.plot_surface(data_x, data_y, data_z, marker="o", cmap='terrain')
        #ax.plot_wireframe(data_x, data_y, data_z)
     #plt.legend(loc = "best")
       
    ax.scatter(xdata , ydata , zdata , marker = "x")
    ax.set_xlabel('input_gain')
    ax.set_ylabel('feed_gain')
    ax.set_zlabel('nmse')
    plt.show()
    #plt.savefig(name + ".png", dpi = 300)
    #plt.cla()

