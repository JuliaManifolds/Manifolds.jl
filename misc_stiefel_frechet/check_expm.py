from os import listdir
import os.path

import h5py
import numpy as np
from scipy.linalg import expm_frechet


def linf(mat):
    return np.max(np.abs(mat))


def check_files():
    data_dir = 'test_expm_frechet'
    fls = listdir(data_dir)
    N = len(fls)
    all_res = np.empty((N, 2))

    def do_one_file(i):
        f = h5py.File(os.path.join(data_dir, fls[i]), 'r')
        A = f['A'][:, :]
        E = f['E'][:, :]
        expA = f['expA'][:, :]
        expAE = f['expAE'][:, :]
        epy = expm_frechet(A, E)
        return linf(epy[0] - expA), linf(epy[1] - expAE)

    for i in range(N):
        all_res[i, :] = do_one_file(i)

    print(np.max(all_res, axis=0))
    # max error is [4.54747351e-12 5.82076609e-11]
