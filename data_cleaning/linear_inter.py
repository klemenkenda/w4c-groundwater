import time

import numpy as np


from data_cleaning.utils import interpolate

orig_full_data = np.load("data/full_np_data.pickle.npy")
orig_dates = np.load("data/dates.pickle.npy")

full_np_data = np.empty(orig_full_data.shape, dtype=float)
full_np_data[:] = np.nan

t = time.time()
for i, data in enumerate(orig_full_data):
    full_np_data[i] = interpolate(data)

print(time.time() - t)

np.save("data/linear_inter.pickle.npy", full_np_data)
print(orig_dates)
t = time.time()
for i, data in enumerate(orig_full_data):
    full_np_data[i] = interpolate(data, 2)

print(time.time() - t)
np.save("data/spline_inter.pickle.npy", full_np_data)

print(orig_dates)
