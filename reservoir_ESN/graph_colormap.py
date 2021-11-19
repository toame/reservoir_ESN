import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np


folder = "nmse_gain_data/"
for name in os.listdir(folder):
    #name = "approx_3_3.0_100"
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.txt', sep=',',comment='#')
    #print(df)
    #p_list = df["p"].unique()
    #input_gain_list = df["input_gain"].unique()
    #feed_gain_list = df["feed_gain"].unique()
    #nmse_list = df["nmse"].unique()
    function_names = ["mackey200"];

    fig, ax = plt.subplots()

    #for function_name in function_names:
    #data_x, data_y, data_z = [], [], []

 
    #array_a = np.genfromtxt( folder + name + '.txt' , skip_header = 1 , delimiter = ',' )

    xdata = df['input_gain']
    ydata = df['feed_gain']
    zdata = df['opt_nmse']

    
    #ax.plot_surface(xdata , ydata , zdata , cmap = 'ocean') 
    ax.scatter(xdata , ydata , c = zdata , marker = "o")
    #ax.plot(xdata , ydata , zdata , marker = "o")
    ax.set_xlabel('input_gain')
    ax.set_ylabel('feed_gain')
    #ax.set_zlabel('opt_nmse')
    cm = plt.cm.get_cmap('RdYlBu')
    mappable = ax.scatter(xdata, ydata, c=zdata, s=35, cmap=cm)
    fig.colorbar(mappable, ax=ax)
    plt.show()
    #plt.savefig(name + ".png", dpi = 300)
    #plt.cla()

