import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('./output_unit/laser_5_0.0_sinc_100_27_4.txt', sep=',',comment='#')
plt.hist(df["input"],  range=(-3, 3), bins=41)
plt.show()
