import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import math
#folder = "output_data/"
folder = "output_data/"
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
    function_names = ["sinc"]
    task_name = "narma10"
    #df_sub = df.query('function_name == "sinc"')
    
    #print(temp)    
    statistics = df.describe()
    statistics.to_csv("statictics.csv")
    #df.to_csv('to_csv_out.csv')
    df2 = df[0:0]
    df2.insert(0, "best_task",0)
    df2.insert(0, "best_task_nmse",0)
    df2.insert(0, "best_task_nmse_sinc",0)
    df2.insert(0, "best_task_nmse_tanh",0)
    print(df)
    for function_name in function_names:
        df_sinc = df.query('function_name == "sinc"')
        df_tanh = df.query('function_name == "tanh"')
        df_sub = df
        temp = df_sub.apply(lambda x: list(x[x == x.min()].index))  
        temp_sinc = df_sinc.apply(lambda x: list(x[x == x.min()].index))  
        temp_tanh = df_tanh.apply(lambda x: list(x[x == x.min()].index))  
        for tau in [0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91]:
            for nu in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]:
                task_name2 = "approx" + str(tau) + "_" + '{:.1f}'.format(nu)
                #task_name2 = "narma" + str(tau)
                if task_name2 not in df.columns:
                    continue
                
                
                df_sub2 = df_sub.loc[temp[task_name2][0]]
                min_nmse = df_sub2[task_name2]
                min_nmse_L = df_sub2.L
                min_nmse_NL =df_sub2.NL
                df_sub_sinc = df_sinc.loc[temp_sinc[task_name2][0]]
                df_sub_tanh = df_tanh.loc[temp_tanh[task_name2][0]]
                min_nmse_sinc = df_sub_sinc[task_name2]
                min_nmse_tanh = df_sub_tanh[task_name2]

                df_sub2["best_task"] = task_name2
                df_sub2["best_task_nmse"] = min_nmse
                df_sub2["best_task_nmse_sinc"] = min_nmse_sinc
                df_sub2["best_task_nmse_tanh"] = min_nmse_tanh
                df2 = df2.append(df_sub2)
                print(df2)
                function_name2 = df_sub2.function_name
                if(function_name2 == "tanh"):
                    marker = "x"
                if(function_name2 == "sinc"):
                    marker = "*"
                #diff = abs(min_nmse_sinc - min_nmse_tanh)
                if(max(min_nmse_sinc,min_nmse_tanh) < 0.01 or min(min_nmse_sinc,min_nmse_tanh)  > 0.9):
                    continue
                    marker = "."
                if(tau == 0):
                    logtau = -1
                else:
                    logtau = math.log2(tau)
                diff = min_nmse_NL + 1e-4
                plt.scatter(nu, logtau,s=200, marker=marker, c = diff, alpha=1.0, linewidths=2, norm=LogNorm(vmin=0.01, vmax=1.0))
                #plt.scatter(min_nmse_L, min_nmse_NL,s=300, marker=marker, alpha=1.0, linewidths=2, label = task_name + "_" + function_name + '{:.3f}'.format(min_nmse))
                print(task_name2)
                print(min_nmse)
    print(df2)
    df2.to_csv("test3.csv")
    plt.ylim(-2.0, 7.0)
    plt.xlim(-4.0, 9.0)
    plt.legend(loc = "best")
    plt.colorbar()
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()
    plt.savefig(name + "2.png", dpi = 600)
    plt.clf()
    break

