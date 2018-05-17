import numpy as np
import os
import pickle
import datetime
import bisect

from math import isnan

full_data = pickle.load(open(os.path.join("data", "parsed.pickle"), "rb"))

min_date = min(data[0][0] for data in full_data)
max_date = max(data[-1][0] for data in full_data)

diff = max_date - min_date
diff_d = diff.days

all_dates = [min_date + datetime.timedelta(days=j) for j in
             range(diff_d + 1)]

print(min_date)
print(max_date)
print(all_dates[0])
print(diff_d)

full_np_data = np.empty((len(full_data), diff_d+1), dtype=float)
full_np_data[:] = np.nan
np_freq = np.empty((len(full_data), 2), dtype=float)

for i, data in enumerate(full_data):
    new = np.empty(diff_d+1)
    new[:] = np.nan
    low = data[0][0]
    low_ind = bisect.bisect(all_dates, low) - 1
    new[low_ind: low_ind + len(data)] = np.array([j[1] for j in data])
    full_np_data[i] = new
    np_freq[i][0] = 1 - (sum(isnan(j[1]) for j in data)/len(data))
    np_freq[i][1] = 1 - (sum(isnan(j) for j in new)/len(new))
    if np_freq[i][0] == 0:
        print("ind:", i)
    assert all_dates[low_ind] == low

np.save("data/full_np_data.pickle", full_np_data)
np.save("data/dates.pickle", np.array(all_dates))
print(np_freq)
