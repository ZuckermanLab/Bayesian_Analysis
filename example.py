# August George - 2021
# An example using MeshGrid.py on a mixed Gaussian model

import MeshGrid as mg
import scipy.stats as stats
import numpy as np
import pandas as pd
import ray
#for debugging
import gc
import psutil


def calc_y(p):
    """
    Example function - 4 Gaussian distributions
    y(x) = c1*N(x|mu1,sigma=1) + c2*N(x|mu2,sigma=1) + c3*N(x|mu3,sigma=1) + c4*N(x|mu4,sigma=1)
    :param p: list of parameters (mean of each Gaussian)
    :return: the calculated pdf based on the parameters
    """

    mu = p[0:4]
    c = p[4:]
    sigma = [2] * len(mu)
    x = np.linspace(-25, 25, 500)
    y = np.zeros(shape=x.shape)
    for i in range(len(mu)):
        y += c[i] * stats.norm.pdf(x, mu[i], sigma[i])
    return y


if __name__ == '__main__':

    ray.init() # for single node
    # ray.init(address='auto')  # for multiple nodes - e.g. cluster using slurm

    # generate synthetic data for example
    p_true = [-10, -5, 5, 10, 0.2, 0.4, 0.6, 0.8]
    sigma_true = 2e-3
    x_true = np.linspace(-25, 25, 500)
    y_true = calc_y(p_true)  # note: you will need to write your own function for your model
    y_obs = y_true + np.random.normal(loc=0, scale=sigma_true, size=np.size(y_true))

    # parameter mesh grid settings
    N = 10  # number of resampled points
    p_input = [('mu_1', -10, 10, 3),
               ('mu_2', -10, 10, 3),
               ('mu_3', -10, 10, 3),
               ('mu_4', -10, 10, 3),
               ('c_1', 0.1, 1, 3),
               ('c_2', 0.1, 1, 3),
               ('c_3', 0.1, 1, 3),
               ('c_4', 0.1, 1, 3),
               ('sigma', 1e-3, 3e-3, 3)]

    mg_df = mg.create_grid_coord(p_input, verbose=True)  # calculate parameter mesh grid

    @ray.remote  # used to run ray remote function - this runs in parallel
    def f(i, df):
        """ This calculates the log likelihood based on the observed data and theta"""
        return mg.calc_logl(y_obs, theta=df.iloc[i], func=calc_y)


    # batch procedure
    b_size = 10000  # how many parameter sets per batch
    mg_df_b = mg_df.groupby(np.arange(len(mg_df)) // b_size)  # note: integer divison for non-even grouping
    b_w = {}  # weights for each batch
    s = []
    for i, mg_b in mg_df_b:
        mg_id = ray.put(mg_b)
        print(f'calculating log-likelihood for batch {i}...')
        logl_ref_list_b = []
        for j in range(len(mg_b.index)):
            logl_ref_list_b.append(f.remote(j, mg_id))  # list (in the same order as meh grid index)
        logl_list_b = ray.get(logl_ref_list_b)  # run ray remote functions
        mg_b['logl'] = logl_list_b  # add logl to dataframe
        score_df = mg.score_grid(mg_b, verbose=True)  # score batch
        start_points, ESS = mg.resample_grid(score_df, N, verbose=True)  # down sample to N
        s.append(start_points)  # add to list of starting point sub samples
        b_w[f'batch {i}'] = start_points['weight'].sum()  # add batch weight


        # debugging: use this to see how much virtual memory is being used and to reset it
        print('\ndebugging:')
        print(f'memory usage {psutil.virtual_memory().percent}')
        print('reseting memory...\n')
        gc.collect()

    b_w_df = pd.DataFrame(list(b_w.items()), columns=['batch n', 'batch weight'])
    b_w_df['rel batch weight'] = b_w_df['batch weight']/b_w_df['batch weight'].sum()

    b_idx = np.random.choice(np.arange(len(b_w_df.index)), size=N, replace=True, p=b_w_df['rel batch weight'])

    starting_points = []
    for i in b_idx:
        ss = s[i]  # select batch based on relative batch weight (b_idx)
        sp = ss.sample()  # randomly sample from group
        starting_points.append(sp)

    print(starting_points)

    # # regular procedure
    # mg_id = ray.put(mg_df)  # ray stores the object id of the mesh grid --> don't have to make copies
    #
    # logl_id_list = []
    # print('calculating log-likelihood...')
    # for i in range(len(mg_df.index)):
    #     logl_id_list.append( f.remote(i, mg_id))  # list (in the same order as meh grid index)
    # logl_list = ray.get(logl_id_list)  # run ray remote functions
    #
    # mg_df['logl'] = logl_list  # new df for log-likelihoods
    # score_df = mg.score_grid(mg_df, verbose=True)  # df relative log-likelihood, likelihood, probability density
    # start_points, ESS = mg.resample_grid(score_df, N, verbose=True)  # df of start points, effective sample size