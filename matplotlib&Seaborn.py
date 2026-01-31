import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 6)  # Set default figure size

data=sns.load_dataset('Penguins')
print(data.head())
