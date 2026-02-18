import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from math import floor
import pandas as pd
import numpy as np
import cvxpy as cp
import sys,os

def price_curves(s, d, lam_u, lam_o):
    """
    Pricing function used for AMM.
    s: Supply
    d: Demand
    lam_u: Feed-in Tarif
    lam_o: Retail price

    Returns
    -------
        r: marginal selling price
        c: marginal buying price
    """
    lam_m = 0.5 * (lam_u + lam_o)
    ratio_sd = np.divide(s, d, out=np.zeros_like(s), where=d > 1e-12)
    ratio_ds = np.divide(d, s, out=np.zeros_like(d), where=s > 1e-12)
    c = lam_m + (lam_o - lam_m) * np.maximum(1 - ratio_sd, 0.0)
    r = lam_m - (lam_m - lam_u) * np.maximum(1 - ratio_ds, 0.0)
    return r, c

def solve_horizon(T, L, B, b0,
                  omega_horizon, alpha_base_horizon,
                  alpha_flex_day_horizon, X, K_max,
                  lam_under, lam_over, gamma_timestep,
                  use_amm=False, s_others_horizon=None, d_others_horizon=None,rC=False):
    """
    A single, unified solver for both AMM and NO AMM scenarios.
    - If use_amm is True, it calculates dynamic AMM prices.
    - If use_amm is False, it uses fixed retailer prices.
    - All constraints and optimization logic are IDENTICAL in both cases.
    """
    # --- 1. SETUP ---
    num_days = L + 1
    H = T * num_days
    time_interval = 24 / T

    # --- 2. PRICING LOGIC (The only part that differs) ---
    if use_amm:
        # Dynamic AMM pricing
        r_h, c_h = price_curves(s_others_horizon, d_others_horizon,
                                np.tile(lam_under, num_days),
                                np.tile(lam_over, num_days))
    else:
        # Fixed retailer pricing
        r_h = np.tile(lam_under, num_days)
        c_h = np.tile(lam_over, num_days)

    # --- 3. CORE OPTIMIZATION (Identical for both scenarios) ---
    discounts = np.array([gamma_timestep**t for t in range(H)])
    r_h_discounted = r_h * discounts
    c_h_discounted = c_h * discounts

    k = cp.Variable(H)
    x_pos = cp.Variable(H)
    x_neg = cp.Variable(H)

    constraints = []
    for t in range(1, H + 1):
        constraints += [cp.sum(k[:t]) <= B - b0, cp.sum(k[:t]) >= -b0]
    constraints += [k <= K_max * time_interval, k >= -K_max * time_interval]
    for t in range(H):
        constraints += [k[t] + x_pos[t] - x_neg[t] <= (omega_horizon[t] - alpha_base_horizon[t]) * time_interval]
    constraints += [x_pos >= 0, x_neg >= 0, x_pos <= X * time_interval, x_neg <= X * time_interval]

    #  Flexible Load Constraint
    for d in range(num_days):
        start, end = d * T, (d + 1) * T
        total_potential_flex_day = np.sum((omega_horizon[start:end] - alpha_base_horizon[start:end]) * time_interval)
        target_for_day = total_potential_flex_day - alpha_flex_day_horizon[d]
        constraints += [cp.sum(k[start:end]) + cp.sum(x_pos[start:end] - x_neg[start:end]) == target_for_day]

    objective = cp.Maximize(r_h_discounted @ x_pos - c_h_discounted @ x_neg)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status in ["infeasible", "unbounded"]:
        return None, None, None, None
    if rC:
        return prob.value, k.value, x_pos.value, x_neg.value, constraints, objective
    return prob.value, k.value, x_pos.value, x_neg.value



def generate_night_owl_profile(base_index, base_load=0.2, peak_load=2.0, seed=123):
    """Generates a demand profile for a night shift worker."""
    np.random.seed(seed)
    # Create a base series
    profile = pd.Series(0.0, index=base_index)
    hours = base_index.hour

    # Define time periods
    is_peak_time = (hours >= 22) | (hours < 6) # 10 PM - 6 AM
    is_shoulder_time = (hours >= 18) & (hours < 22) # 6 PM - 10 PM (ramping up)
    is_day_sleep = (hours >= 9) & (hours < 17) # 9 AM - 5 PM (minimal)

    # Assign load based on time
    profile[is_peak_time] = peak_load
    profile[is_shoulder_time] = peak_load / 2
    profile[is_day_sleep] = 0.05 # Just the fridge

    # Add seasonality (more heating/lighting in winter)
    day_of_year = base_index.dayofyear
    seasonal_factor = 1 + 0.4 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)

    # Combine with base load, noise, and seasonality
    noise = np.random.rand(len(base_index)) * 0.2
    final_profile = (base_load + profile + noise) * seasonal_factor

    return pd.DataFrame({'demand_kw': final_profile})

def generate_remote_worker_profile(base_index, base_load=0.3, work_load=1.2, seed=456):
    """Generates a demand profile for a remote worker."""
    np.random.seed(seed)
    profile = pd.Series(0.0, index=base_index)
    hours = base_index.hour

    # Define time periods
    is_work_time = (hours >= 9) & (hours < 17)
    is_evening = (hours >= 17) & (hours < 23)

    # Assign load
    profile[is_work_time] = work_load
    profile[is_evening] = work_load / 2

    # Add a small morning ramp-up
    morning_ramp = (hours >= 7) & (hours < 9)
    profile[morning_ramp] = 0.5

    # Seasonality
    day_of_year = base_index.dayofyear
    seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 25) / 365.25)

    noise = np.random.rand(len(base_index)) * 0.15
    final_profile = (base_load + profile + noise) * seasonal_factor

    return pd.DataFrame({'demand_kw': final_profile})

def generate_commuter_profile(base_index, base_load=0.2, morning_peak=1.5, evening_peak=2.5, seed=789):
    """Generates a classic double-hump commuter demand profile."""
    np.random.seed(seed)
    profile = pd.Series(0.0, index=base_index)
    hours = base_index.hour

    # Define peaks
    is_morning_peak = (hours >= 6) & (hours < 9)
    is_evening_peak = (hours >= 17) & (hours < 22)

    # Assign load
    profile[is_morning_peak] = morning_peak
    profile[is_evening_peak] = evening_peak

    # Seasonality
    day_of_year = base_index.dayofyear
    seasonal_factor = 1 + 0.5 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)

    noise = np.random.rand(len(base_index)) * 0.25
    final_profile = (base_load + profile + noise) * seasonal_factor

    return pd.DataFrame({'demand_kw': final_profile})
