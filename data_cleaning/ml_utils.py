import datetime
import os
import pickle

import numpy as np
from bisect import bisect


def find_date_ind(date, dates):
    return bisect(dates, date) - 1


def available_indices(dates, start_date: datetime.date, end_date: datetime.date,
                      data, flt=True):
    if not flt:
        return list(range(len(data)))
    start_ind = find_date_ind(start_date, dates)
    end_ind = find_date_ind(end_date, dates)
    rtr = []
    for j, data in enumerate(data):
        if not np.isnan(data[start_ind]) and not np.isnan(data[end_ind - 1]):
            rtr.append(j)

    return rtr


def extract_data(dates, start_date: datetime.date, end_date: datetime.date,
                 data, flt=True):
    indices = available_indices(dates, start_date, end_date, data, flt)
    start_ind = find_date_ind(start_date, dates)
    end_ind = find_date_ind(end_date, dates)

    return indices, [j[start_ind:end_ind] for j in data[np.array(indices)]], {"start": start_ind, "end": end_ind}


def make_bins(data, bin_size, method=np.nanmean):
    # print(len(data))
    full_len = (len(data) // bin_size) * bin_size
    if full_len != len(data):
        dst_len = full_len + bin_size
    else:
        dst_len = len(data)
    dat = np.pad(data, (0, dst_len - len(data)), "constant",
                 constant_values=(0, np.nan))

    dat = dat.reshape((len(dat) // bin_size, bin_size))
    rtr = np.apply_along_axis(method, 1, dat)
    return rtr


def get_sliding_data(dates, start_date: datetime.date, end_date: datetime.date,
                     data, bin_size=7, method=np.nanmean):
    _, dat = extract_data(dates, start_date, end_date, data, flt=False)
    dat = np.array([make_bins(sample, bin_size, method) for sample in dat])
    indices = []
    for i, sample in enumerate(dat):
        if not np.isnan(sample[0]) and not np.isnan(sample[-1]):
            indices.append(i)
    return indices, dat[indices]


def main():
    start_date = datetime.date(day=3, month=2, year=2007)
    end_date = datetime.date(day=3, month=12, year=2008)
    print(make_bins(np.array(list(range(15)), dtype=float), 5))

    DATA_FOLDER = os.path.join("data")
    def path(filename):
        return os.path.join(DATA_FOLDER, filename)

    orig_full_data = np.load(path("full_np_data.pickle.npy"))
    inter_full_data = np.load(path("linear_inter.pickle.npy"))
    spline_full_data = np.load(path("spline_inter.pickle.npy"))
    orig_dates = np.load(path("dates.pickle.npy"))
    names = pickle.load(open(path("filenames.pickle"), "rb"))
    dates = np.load(path("dates.pickle.npy"))
    strange = np.array([orig_full_data[4]])
    a, b = get_sliding_data(orig_dates, start_date, end_date, strange)
    print(a)

if __name__ == '__main__':
    main()
