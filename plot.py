# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:15:15 2023

@author: jayaraman
"""

import matplotlib.pyplot as plt
import pandas


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load data from a CSV file using pandas
# Replace 'your_data.csv' with the path to your CSV file
df = pd.read_csv('state-transient-response.csv')
#%%
# Step 2: Explore and analyze the data
# You can print the first few rows of the DataFrame to get an overview of the data
print(df.head())

# Step 3: Plot the data
# Let's assume your CSV file has columns 'x' and 'y' that you want to plot
x = df['x']
y = df['Curve1']

#%% smooth curve
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt



window_size = 9  # Window size must be odd
order = 2  # Polynomial order

smoothed = savgol_filter(y, window_size, order)
smoothed2 = savgol_filter(smoothed, window_size, order)

plt.plot(x, smoothed2)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('Voltage')
plt.title('Transient response')
plt.legend()
plt.show()


#%%
# Create a basic line plot
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.plot(x, y)  # 'o' adds data points as markers

# Add labels and a title


# Add a legend (if multiple lines are present)
plt.legend()

# Display the plot (if you're not in a Jupyter Notebook)
plt.show()

#%% step signal

df2 = pandas.DataFrame(columns=["time", "current"])
time = []
t1 = np.linspace(99.5, 100.2, 100)
t2= 100*[100.2]
t3 = np.linspace(100.2, 100.4, 100)
t4= 100*[100.4]
t5 = np.linspace(100.4, 100.6, 100)
t6 = 100*[100.6]
t7 = np.linspace(100.6, 100.7, 100)
t8 = 100*[100.7]
t9 = np.linspace(100.7, 102, 100)

time =np.append(time, t1)

time =np.append(time, t2)
time =np.append(time, t3)
time =np.append(time, t4)
time =np.append(time, t5)
time =np.append(time, t6)
time =np.append(time, t7)
time =np.append(time, t8)
time =np.append(time, t9)

current = []
c1= 100*[12.5]
c2 = np.linspace(12.5, 20.5, 100)
c3 = 100*[20.5]
c4 = np.linspace(20.5, 24.5, 100)
c5 = 100*[24.5]
c6 = np.linspace(24.5, 20.5, 100)
c7 = 100*[20.5]
c8 = np.linspace(20.5, 13, 100)
c9 = 100*[13]
current = np.append(current, c1)
current =np.append(current, c2)
current =np.append(current, c3)
current =np.append(current, c4)
current =np.append(current, c5)
current =np.append(current, c6)
current =np.append(current, c7)
current =np.append(current, c8)
current =np.append(current, c9)

df2["current"] = current
df2["time"] = time
plt.plot(time, current)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('current')
plt.title('Transient input')
plt.legend()
plt.show()