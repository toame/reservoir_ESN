import pandas as pd
import os
from matplotlib import pyplot as plt
folder = "output_data/"
for name in os.listdir(folder):
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.csv', sep=',',comment='#')
    function_names = ["sinc", "tanh"]
    for function_name in function_names:
        data_x = df[(df.function_name == function_name)].L
        data_y = df[(df.function_name == function_name)].NL
        plt.scatter(data_x, data_y,s=10, marker="o", alpha=0.3, linewidths=1, label = function_name)
    plt.legend(loc = "best")
    #plt.show()
    plt.savefig(name + ".png", dpi = 300)
    plt.cla()

