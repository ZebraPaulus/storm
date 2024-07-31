import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

times = pd.read_csv('times.csv')
# convert absolute times to relative time from last time in row
times['end'] = times['end'] - times['set_lines']
times['set_lines'] = times['set_lines'] - times['set_gym']
times['set_gym'] = times['set_gym'] - times['get_error']
times['get_error'] = times['get_error'] - times['get_command']
times['get_command'] = times['get_command'] - times['update_params']
times['update_params'] = times['update_params'] - times['get_pose']
times['get_pose'] = times['get_pose'] - times['step']
times['step'] = times['step'] - times['start']
times['start'] = 0
print(times)
# filter out all rows with a 95% percentile high value in a column
times = times[times["get_command"] < times["get_command"].quantile(0.94)]
times = times[times["get_error"] < times["get_error"].quantile(0.99)]
for col in times.columns:
    if times[col].quantile(0.95) > 0.003:
        print("[Percentiles]: ",col.ljust(15), times[col].quantile(0.99),",\t", times[col].quantile(0.95),",\t", times[col].quantile(0.75))
        print("[Mean]: ".rjust(15),col.ljust(15), times[col].mean())
# make boxplots of the times
plt.figure(figsize=(10,10))
# make 9 subplots
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.boxplot(times.iloc[:,i],)
    plt.ylabel('Time (s)')
    plt.xticks([1], [times.columns[i]])
    plt.grid()
plt.title('Boxplot of times')
plt.savefig('times_boxplot.png')