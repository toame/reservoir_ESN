import pandas as pd
import os
from matplotlib import pyplot as plt
folder = "Various_ESN_comparison/"
for name in os.listdir(folder):
    #name = "approx_3_3.0_100"
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.txt', sep=',',comment='#')
    p_list = df["p"].unique()
    function_names = ["TD_MG", "TD_ikeda","RND_sinc"];
    for function_name in function_names:
        data_x, data_y = [], []
        for p in p_list:
            data_x.append(p)
            data_y.append(df[(df.function_name == function_name) & (df.p == p)].test_nmse.mean())
        #plt.ylim(0, 0.3)
        plt.yscale("log")
        plt.plot(data_x, data_y, marker="o", label = function_name)
    plt.xlabel("p",size=12)
    plt.ylabel("NMSE",size=12)
    plt.legend(loc = "best")
    plt.subplots_adjust(left=0.15,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    #plt.tight_layout()
    #plt.figure(figsize=(20,20))
    #plt.show()
    plt.savefig(name + ".png", dpi = 300)
    plt.cla()

