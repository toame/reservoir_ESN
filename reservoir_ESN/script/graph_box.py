import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns

folder = "output_data/"
for name in os.listdir(folder):
    #name = "approx_3_3.0_100"
    name = os.path.splitext(os.path.basename(name))[0]
    df = pd.read_csv(folder + name + '.txt', sep=',',comment='#')
    plt.yscale("log")
    df = df.sort_values('p')
    sns.boxplot(x='p', y='nmse', hue='function_name', data=df)
    plt.savefig(name + ".png", dpi = 500)
    plt.cla()

