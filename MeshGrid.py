# August George - 2021
import numpy as np
import pandas as pd
import scipy.stats as stats
import ray


def calc_y(p):
    """
    Example function - 4 Gaussian distributions
    y(x) = c1*N(x|mu1,sigma=1) + c2*N(x|mu2,sigma=1) + c3*N(x|mu3,sigma=1) + c4*N(x|mu4,sigma=1)
    :param p: list of parameters (mean of each Gaussian)
    :return: the calculated pdf based on the parameters
    """

    mu = p[0:4]
    c = p[4:]
    sigma = [2]*len(mu)
    x = np.linspace(-25, 25, 500)
    y = np.zeros(shape=x.shape)
    for i in range(len(mu)):
        y += c[i]*stats.norm.pdf(x, mu[i], sigma[i])
    return y


def calc_logl(y_obs, theta):
    """
    Calculates the (normal) log-likelihood
    :param y_obs: array of observed data
    :param theta: array of parameters - a single parameter set - SIGMA at the end
    :return: the log-likelihood probability
    """

    p = theta[:-1]
    sigma = theta[-1]
    y_pred = calc_y(p)
    # Note: Normal log likelihood = -(n/2)ln(2*pi) -(n/2)ln(sigma^2) -(1/(2simga^2))SUM(x_i-mu)^2
    logp = -len(y_obs) * np.log(np.sqrt(2.0 * np.pi) * sigma)
    logp += -np.sum((y_obs - y_pred) ** 2.0) / (2.0 * sigma ** 2.0)
    return logp


def create_grid_coord(p_input, verbose=True):
    """
    Creates a mesh grid in parameter space from parameter vectors
    :param p_input: a list of tuples to generate a parameter vector
    [parameter name, lower value, upper value, number of divisions]
    :param verbose: a boolean which turns on/off text output
    :return: a dataframe where row_i is parameter set_i and col_i is parameter_i.
    Note: includes a label row and an index column
    """

    p = []
    p_names = []
    if verbose:
        print('creating parameter vectors...')

    for i, p_i in enumerate(p_input):
        p_i_coord = np.linspace(p_i[1], p_i[2], int(p_i[3]))
        if verbose:
            print(f'{p_i[0]}: {p_i_coord}')
        p.append(p_i_coord)
        p_names.append(p_i[0])
    if verbose:
        print('creating parameter space grid coordinates...')
    g = np.meshgrid(*p)
    coord_df = pd.DataFrame(np.vstack(list(map(np.ravel, g))).T, columns=p_names)
    if verbose:
        print(f'{coord_df}')
    return coord_df


def score_grid(grid, verbose=True):
    """
    Calculates relative log-likelihood, likelihood, probability, and effective sample size
    :param grid: a dataframe where row_i is parameter set_i and col_i is parameter_i. must also contain 'logl' col
    :param verbose: a boolean which turns on/off text output
    :return: an updated 'grid' dataframe containing additional columns for log-likelihood, likelihood, and probability
    """

    if verbose:
        print('scoring grid...')
    grid['rel logl'] = np.nan
    grid['rel logl'] = (grid['logl'] - np.max(grid['logl']))
    grid['like'] = np.exp(grid['rel logl'])
    grid['p(x)'] = grid['like']/np.sum(grid['like'])
    if verbose:
        print(f'{grid}')
    return grid


def resample_grid(scored_grid, M, verbose=True):
    """
    Resample from the dataframe of parameter sets based on likelihood probabilities
    :param scored_grid: a dataframe containing a 'p(x)' column of probability densities (sum p = 1)
    :param M: integer number of resamples to take
    :param verbose: a boolean which turns on/off text output
    :return: a filtered version of the scored grid containing only the resampled values, effective sample size
    """

    if verbose:
        print('resampling grid...')
    start_idx = np.random.choice(np.arange(len(scored_grid.index)), size=M, replace=True, p=scored_grid['p(x)'])
    start_p_sets = scored_grid.iloc[start_idx]
    ess = np.sum(scored_grid['p(x)']) / np.max(scored_grid['p(x)'])
    if verbose:
        print(start_p_sets)
        print(f'ESS estimate: {ess}')
    return (start_p_sets, ess)


if __name__ == '__main__':

    ray.init() # for single node
    # ray.init(address='auto')  # for multiple nodes - e.g. cluster using slurm

    # generate synthetic data for example
    p_true = [-10, -5, 5, 10, 0.2, 0.4, 0.6, 0.8]
    sigma_true = 2e-3
    x_true = np.linspace(-25, 25, 500)
    y_true = calc_y(p_true)
    y_obs = y_true + np.random.normal(loc=0, scale=sigma_true, size=np.size(y_true))

    # parameter mesh grid settings
    N = 10  # number of resampled points
    p_input = [('mu_1', -10, 10, 5),
               ('mu_2', -10, 10, 5),
               ('mu_3', -10, 10, 5),
               ('mu_4', -10, 10, 5),
               ('c_1', 0.1, 1, 5),
               ('c_2', 0.1, 1, 5),
               ('c_3', 0.1, 1, 5),
               ('c_4', 0.1, 1, 5),
               ('sigma', 1e-3, 3e-3, 5)]

    mg_df = create_grid_coord(p_input, verbose=True)  # calculate parameter mesh grid

    ### start logl parallelization
    mg_id = ray.put(mg_df)  # ray stores the object id of the mesh grid --> don't have to make copies

    @ray.remote  # used to run ray remote function - this is run in parallel
    def f(i, df):
        """ This calculates the log likelihood based on the observed data and theta"""
        return calc_logl(y_obs, theta=df.iloc[i] )

    logl_id_list = []
    for i in range(len(mg_df.index)):
        logl_id_list.append( f.remote(i, mg_id))  # list (in the same order as meh grid index)
    logl_list = ray.get(logl_id_list)
    mg_df['logl'] = logl_list  # new df for log-likelihoods
    ### end logl parallelization

    print(mg_df)

    score_df = score_grid(mg_df, verbose=True)  # df relative log-likelihood, likelihood, probability density
    start_points, ESS = resample_grid(score_df, N, verbose=True)  # df of start points, effective sample size


