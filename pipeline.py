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

def get_datum(options, S, T, R):
    # R = rates[(rates['date'] == date) & (rates['days'] == rate_dist)]['rate'][0] / 100
    processed = options[['strike_price','best_bid','best_offer']]
    processed['vol_high'] = processed.apply(lambda x: implied_vol(x['best_offer'], S, x['strike_price'], T, R), axis=1)
    processed['vol_low'] = processed.apply(lambda x: implied_vol(x['best_bid'], S, x['strike_price'], T, R), axis=1)
    processed['vol_mid'] = (processed['vol_high'] + processed['vol_low']) / 2
    boundaries = processed['strike_price'].apply(np.log).to_numpy()
    log_strikes = processed['strike_price'].apply(np.log).to_numpy()
    inp_data = torch.tensor(np.vstack([processed['vol_high'].to_numpy().reshape(-1,1), processed['vol_low'].to_numpy().reshape(-1,1), log_strikes.reshape(-1,1)]))
    S = torch.tensor(S).reshape(1,1)
    T = torch.tensor(T).reshape(1,1)
    R = torch.tensor(R).reshape(1,1)
    inp_data = torch.vstack([inp_data, S, T, R])
    print(inp_data)
    return inp_data, log_strikes, 