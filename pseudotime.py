import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


# Helper Functions

def distance(t):
    '''
    Squared distance matrix of t.
    @param t Array of pseudotimes of length N
    '''
    n = len(t)
    D = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(i + 1, n):
            D[i][j] = (t[i] - t[j]) ** 2
    return D + np.transpose(D)


def cross_distance(t1, t2):
    '''
    Squared distance matrix between t1 and t2.
    @param t1, t2 Arrays of pseudotimes
    '''
    n1 = len(t1)
    n2 = len(t2)
    D = np.zeros((n1, n2))
    for i in xrange(n1):
        for j in xrange(n2):
            D[i][j] = (t1[i] - t2[j]) ** 2
    return D


def covariance_matrix(t, lambda_):
    '''
    Covariance matrix K_j of t using the squared exponential covariance function.
    @param t Array of pseudotimes of length N
    @param lambda_ Value of lambda_j
    '''
    D = distance(t)
    return np.exp(-D / (2 * lambda_ ** 2))


def cross_covariance_matrix(t1, t2, lambda_):
    '''
    Covariance matrix K*_j between t1 and t2 using the squared exponential covariance function.
    @param t1, t2 Arrays of pseudotimes
    @param lambda_ Value of lambda_j
    '''
    D = cross_distance(t1, t2)
    return np.exp(-D / (2 * lambda_ ** 2))


def log_likelihood(X, t, lambda_, sigma):
    '''
    Log likelihood of P(X | t, lambda_, sigma), defined by a multivariate Gaussian with mean 0 and covariance K + sigma * I.
    @param X Data array for N points in P dimensions
    @param t Array of pseudotimes of length N
    @param lambda_ Array of lambdas of length P
    @param sigma Array of sigmas of length P
    '''
    n = len(t)
    likelihood = 0
    for i in xrange(X.shape[1]):
        likelihood += stats.multivariate_normal.logpdf(X[:,i], mean=np.zeros(n), cov=covariance_matrix(t, lambda_[i]) + sigma[i] * np.identity(n))
    return likelihood


def sample(mean, var, lower=0, upper=float("inf")):
    '''
    Sample values from a truncated normal of (mean, var) in the range [lower, upper].
    @param mean Array of means
    @param var Array of variances
    @param lower Lower bound
    @param upper Upper bound
    '''
    a = (lower - mean) / var
    b = upper if upper == float("inf") else (upper - mean) / var
    return stats.truncnorm.rvs(a, b, loc=mean, scale=var)


# Priors

def corp_prior(t, r=1):
    '''
    Coulomb repulsive process (Corp) prior on pseudotimes.
    @param Array of pseudotimes of length N
    @param r Repulsion parameter
    '''
    # uniform prior
    if r == 0: return 0

    likelihood = 0
    n = len(t)
    for i in xrange(n):
        for j in xrange(i + 1, n):
            likelihood += math.log(math.sin(math.pi * abs(t[i] - t[j])))
    return 2 * r * likelihood


def lambda_prior(lambda_, scale=1.):
    '''
    Exponential prior on lambdas.
    @param lambda_ Array of lambdas of length P
    @param scale Scale of exponential distribution
    '''
    return sum(stats.expon.logpdf(lambda_, scale=scale))


def sigma_prior(sigma, alpha=1., scale=1.):
    '''
    Inverse gamma prior on sigmas.
    @param sigma Array of sigmas of length P
    @param alpha Shape of inverse gamma distribution
    @param scale Scale of inverse gamma distribution
    '''
    return sum(stats.invgamma.logpdf(sigma, alpha, scale=scale))


# Metropolis Hastings

def acceptance_ratio(X, t_new, t, lambda_new, lambda_, sigma_new, sigma, r):
    '''
    Computes the acceptance ratio, which is defined as posterior_new/posterior_old.
    @param X Data array for N points in P dimensions
    @param t_new New pseudotimes of length N
    @param t Previous pseudotimes of length N
    @param lambda_new New lambdas of length P
    @param lambda_ Previous lambdas of length P
    @param sigma_new New sigmas of length P
    @param sigma Previous sigmas of length P
    @param r Corp parameter
    '''
    likelihood = log_likelihood(X, t_new, lambda_new, sigma_new) - log_likelihood(X, t, lambda_, sigma)
    t_prior = corp_prior(t_new, r) - corp_prior(t, r)
    l_prior = lambda_prior(lambda_new) - lambda_prior(lambda_)
    s_prior = sigma_prior(sigma_new) - sigma_prior(sigma)
    return likelihood + t_prior + t_prior + s_prior


# GPLVM

def GPLVM(X, n_iter, burn, thin, t, t_var, lambda_, lambda_var, sigma, sigma_var, r=1, return_burn=False):
    '''
    Runs the GPLVM.
    @param X Data array for N points in P dimensions
    @param n_iter Number of iterations
    @param burn Burn-in period
    @param thin Thinning parameter
    @param t, lambda_, sigma Initial values for the Markov Chain
    @param t_var, lambda_var, sigma_var Variances for the proposed distributions
    @param r Corp parameter
    @param return_burn If the burn-in period of the traces should be returned
    '''
    n, p = X.shape
    chain_size = int(n_iter/thin) # size of thinned chain
    burn_thin = int(burn/thin) # size of burn region in thinned chain

    # initialize chains
    t_chain = np.zeros((chain_size, n))
    t_chain[0,:] = t
    lambda_chain = np.zeros((chain_size, p))
    lambda_chain[0,:] = lambda_
    sigma_chain = np.zeros((chain_size, p))
    sigma_chain[0,:] = sigma
    likelihood_chain = np.zeros(chain_size)
    likelihood_chain[0] = log_likelihood(X, t, lambda_, sigma)
    prior_chain = np.zeros(chain_size)
    prior_chain[0] = corp_prior(t, r)
    accepted = np.zeros(n_iter)

    # Metropolis Hastings
    for i in xrange(n_iter):
        # sample new t, lambda_, sigma
        t_new = sample(t, t_var, 0, 1)
        lambda_new = sample(lambda_, lambda_var)
        sigma_new = sample(sigma, sigma_var)

        # calculate acceptance ratio
        acceptance = acceptance_ratio(X, t_new, t, lambda_new, lambda_, sigma_new, sigma, r)

        if acceptance > math.log(np.random.rand()):
            # accept
            accepted[i] = 1
            t = t_new
            lambda_ = lambda_new
            sigma = sigma_new

        if i % thin == 0:
            # update traces
            j = i / thin
            t_chain[j,:] = t
            lambda_chain[j,:] = lambda_
            sigma_chain[j,:] = sigma
            likelihood_chain[j] = log_likelihood(X, t, lambda_, sigma)
            prior_chain[j] = corp_prior(t, r)

    burn_idx = 0 if return_burn else burn_thin

    return {
        "t_chain" : t_chain[burn_idx:,:],
        "lambda_chain" : lambda_chain[burn_idx:,:],
        "sigma_chain" : sigma_chain[burn_idx:,:],
        "acceptance_rate" : sum(accepted)/len(accepted),
        "burn_acceptance_rate" : sum(accepted[burn_idx:]/len(accepted[burn_idx:])),
        "r" : r,
        "likelihood_chain" : likelihood_chain,
        "prior_chain" : prior_chain,
        "params" : {
            "n_iter" : n_iter,
            "burn" : burn,
            "thin" : thin,
            "burn_idx" : burn_idx
        }
    }


# Posterior Predictions

def predict(X, t_vals, t_avgs, lambda_avgs, sigma_avgs):
    '''
    Returns the posterior mean estimate X_p, where X_p[:,j] = K*_j * K^-1_j * X[:,j].
    @param X Data array for N points in P dimensions
    @param t_vals Pseudotime values at which to predict
    @param t_avgs, lambda_avgs, sigma_avgs Averages from the Metropolis Hastings walk
    '''
    n = len(t_vals)
    p = X.shape[1]

    X_p = np.zeros((n, p))
    for i in xrange(p):
        K_star = cross_covariance_matrix(t_vals, t_avgs, lambda_avgs[i])
        K = covariance_matrix(t_avgs, lambda_avgs[i])
        X_p[:,i] = K_star * np.linalg.inv(K) * X[:,i]

    return X_p


# Plotting Functions

def plot_pseudotime_trace(gplvm, n, samples=10):
    if samples == n:
        cols = np.arange(n)
    else:
        cols = np.random.choice(n, samples, replace=False)
    params = gplvm["params"]
    df = pd.DataFrame(gplvm["t_chain"][:,cols], index=np.arange(params["burn_idx"], params["n_iter"], params["thin"]))
    df.plot(legend=False)
    sns.despine()
    plt.show()


###############################################################################

# Seaborn Setup
sns.set_style("white")


# Synthetic Data
np.random.seed(1246)
n = 30
p = 2
lambda_ = np.array([1/math.sqrt(2)] * p)
sigma = np.array([1e-3] * p)
t_real = np.random.sample(n)
X = np.zeros((n, p))
for i in xrange(p):
    X[:,i] = stats.multivariate_normal.rvs(mean=np.zeros(n), cov=covariance_matrix(t_real, lambda_[i]) + sigma[i] * np.identity(n))

# plot X
# plt.scatter(X[:,0], X[:,1], c=t_real)
# sns.despine()
# plt.show()

n_iter = 3000
burn = 0
thin = 60

t = stats.uniform.rvs(loc=.499, scale=.002, size=n)
t_var = np.array([.5e-3] * n)
lambda_var = np.array([.5e-5] * p)
sigma_var = np.array([.5e-10] * p)

gplvm = GPLVM(X, n_iter, burn, thin, t, t_var, lambda_, lambda_var, sigma, sigma_var)
print gplvm["acceptance_rate"]

plot_pseudotime_trace(gplvm, n, n)