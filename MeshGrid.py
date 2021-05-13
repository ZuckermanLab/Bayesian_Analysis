# August George - 2021
# Collected scripts to help create a mesh grid in parameter space
import numpy as np
import pandas as pd

### NOT CURRENTLY USED
def calc_logl(y_obs, theta, func):
    """
    NOT CURRENTLY USED Calculates the (normal) log-likelihood
    :param y_obs: array of observed data
    :param theta: array of parameters - a single parameter set - SIGMA at the end
    :param func: function which inputs a list of parameters and outputs the predicted value
    :return: the log-likelihood probability
    """

    p = theta[:-1]
    sigma = theta[-1]
    y_pred = func(p)
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
    :return: an updated 'grid' dataframe containing additional columns for rel log-likelihood, likelihood, and weights
    """

    if verbose:
        print('scoring grid...')
    grid['rel logl'] = (grid['logl'] - np.max(grid['logl']))  # log(p/p_max) --> log(p) -max(log(p))
    grid['rel like'] = np.exp(grid['rel logl'])  # log(p) --> p = exp(log(p))
    grid['weight'] = grid['rel like']/np.sum(grid['rel like'])  # sum(p) = 1 for resampling
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
    start_idx = np.random.choice(np.arange(len(scored_grid.index)), size=M, replace=True, p=scored_grid['weight'])
    start_p_sets = scored_grid.iloc[start_idx]
    ess = np.sum(scored_grid['rel like'])  # ESS = sum(likelihood)/max
    if verbose:
        print(start_p_sets)
        print(f'ESS estimate: {ess}')
    return (start_p_sets, ess)


if __name__ == '__main__':
    pass
