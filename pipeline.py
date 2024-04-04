import torch
import numpy as np
from scipy.stats import norm
N = norm.cdf

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(target_value, S, K, T, r, PRECISION = 1.0e-9, MAX_ITERATIONS = 200):
    sigma = 0.5
    # print('Params:',S, K, T, r)
    # print('Target:', target_value)
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        # print('Price on it %s with sigma %s: %s' % (i, sigma, price))
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        # print('Diff',diff)
        if (abs(diff) < PRECISION):
            return sigma
        sigma = max(sigma + diff/vega, PRECISION) # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far
# Copied above ^^

# Load data

