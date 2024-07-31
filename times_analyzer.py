import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

times = pd.read_csv('times.csv')
# get time deltas
for i in range(len(times.columns)-1,0,-1):
    times[times.columns[i]] = times[times.columns[i]]-times[times.columns[i-1]]
# set first column as sum of all other columns
times[times.columns[0]] = 0
times[times.columns[0]] = times.sum(axis=1)

print(times)
# filter out all rows with a 95% percentile high value in a column
print("\nTiming overview:\n"+"="*60)
for col in times.columns:
    if times[col].quantile(0.95) > 0.003:
        print("[Percentiles]: ".rjust(18)+"{}: {:.5f} | {:.5f} | {:.5f} |".format(col.ljust(11), times[col].quantile(0.99), times[col].quantile(0.95), times[col].quantile(0.75)))
        print("[Mean, Max, Min]: {}: {:.5f} | {:.5f} | {:.5f} |".format(col.ljust(11), times[col].mean(), times[col].max(), times[col].min()))
        print("-"*60)

# make boxplots of the times
plt.figure(figsize=(10,10))
# make 8 subplots
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.boxplot(times.iloc[:,i],)
    plt.ylabel('Time (s)')
    plt.xticks([1], [times.columns[i]])
    plt.grid()
plt.title('Boxplot of times')
plt.savefig('times_boxplot.png')