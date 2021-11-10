import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('./output_unit/tanh_0_6.txt', sep=',',comment='#')
plt.hist(df["input"],  range=(-3.0, 3.0), bins=41)
plt.show()
