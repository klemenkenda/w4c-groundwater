import datetime

import numpy as np
from bisect import bisect


def find_date_ind(date, dates):
    return bisect(dates, date) - 1


def available_indices(dates, start_date: datetime.date, end_date: datetime.date, data):
    start_ind = find_date_ind(start_date, dates)
    end_ind = find_date_ind(end_date, dates)
    rtr = []
    for j, data in enumerate(data):
        if not np.isnan(data[start_ind]) and not np.isnan(data[end_ind-1]):
            rtr.append(j)

    return rtr


def extract_data(dates, start_date: datetime.date, end_date: datetime.date, data):
    indices = available_indices(dates, start_date, end_date, data)
    start_ind = find_date_ind(start_date, dates)
    end_ind = find_date_ind(end_date, dates)

    return indices, [j[start_ind:end_ind] for j in data[np.array(indices)]]


def main():
    start_date = datetime.date(day=3, month=1, year=1984)
    end_date = datetime.date(day=3, month=1, year=1994)

if __name__ == '__main__':
    main()
