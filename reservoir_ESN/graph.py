import pandas as pd
import os
from matplotlib import pyplot as plt
folder = "output_data/"
for name in os.listdir(folder):
    #name = "approx_3_3.0_100"
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.txt', sep=',',comment='#')
    p_list = df["p"].unique()
    function_names = ["sinc", "tanh", "gauss", "oddsinc"]
    for function_name in function_names:
        data_x, data_y = [], []
        for p in p_list:
            data_x.append(p)
            data_y.append(df[(df.function_name == function_name) & (df.p == p)].test_nmse.mean())
        #plt.ylim(0, 0.3)
        plt.yscale("log")
        plt.plot(data_x, data_y, marker="o", label = function_name)
    plt.legend(loc = "best")
    #plt.show()
    plt.savefig(name + ".png", dpi = 300)
    plt.cla()

