import numpy as np

f = open("times.txt","r")
temp = np.array([float(l) for l in f])
f.close()
times = [temp[i]-temp[i-1] for i in range(1,len(temp))]
print("Mean time: ", np.mean(times))
print("Median time: ", np.median(times))
print("75th percentile: ", np.percentile(times, 75))
print("95th percentile: ", np.percentile(times, 95))
print("Duration: ", temp[-1]-temp[0])
