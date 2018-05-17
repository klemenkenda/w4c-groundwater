import numpy as np
from scipy.interpolate import splrep, splev


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def linear_interpolate(y):
    nans, x = nan_helper(y)
    rtr = np.copy(y)
    rtr[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return rtr


def find_fst_non_nan(x):
    return np.argmax(~np.isnan(x))


def find_last_non_nan(x):
    return len(x) - find_fst_non_nan(np.flipud(x))


def spline_interpolate(y):
    xx = np.linspace(0, len(y), len(y))
    nans, x = nan_helper(y)
    tck = splrep(xx[~nans], y[x(~nans)])
    rtr = splev(xx, tck)
    return rtr

def interpolate(x, method=1):
    fst = find_fst_non_nan(x)
    last = find_last_non_nan(x)
    rtr = np.empty(len(x), dtype=float)
    rtr[:] = np.nan
    if method == 1:
        rtr[fst:last] = linear_interpolate(x[fst:last])
    else:
        rtr[fst:last] = spline_interpolate(x[fst:last])
    return rtr


def main():
    a = np.array([0, float("Nan"), 2, float("Nan"), 4, 5, float("Nan"), 7])

    print(np.sin(a))
    print(list(spline_interpolate(np.sin(a))))

if __name__ == '__main__':
    main()
