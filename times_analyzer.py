import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

times = pd.read_csv('times.csv')
# get time deltas
for i in range(len(times.columns)-1,0,-1):
    times[times.columns[i]] = times[times.columns[i]]-times[times.columns[i-1]]
# drop starting time
times.drop(columns=[times.columns[0]], inplace=True)

print(times)
# filter out all rows with a 95% percentile high value in a column
for col in times.columns:
    if times[col].quantile(0.95) > 0.003:
        print("[Percentiles]: ".rjust(18),col.ljust(15), times[col].quantile(0.99),",\t", times[col].quantile(0.95),",\t", times[col].quantile(0.75))
        print("[Mean, Max, Min]: ",col.ljust(15), times[col].mean(), ",\t", times[col].max(), ",\t", times[col].min())

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