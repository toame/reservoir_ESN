import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('./output_data/laser_6_0.0_100.txt', sep=',',comment='#')
p_list = df["p"].unique()
function_names = ["sinc", "tanh", "gauss", "oddsinc"]
for function_name in function_names:
    data_x, data_y = [], []
    for p in p_list:
        data_x.append(p)
        data_y.append(df[(df.function_name == function_name) & (df.p == p)].test_nmse.mean())
    plt.ylim(0, 0.3)
    plt.plot(data_x, data_y, marker="o", label = function_name)
plt.legend(loc = "best")
plt.show()

