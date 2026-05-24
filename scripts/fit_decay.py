#!/usr/bin/env python3
"""
Research Analysis Script: Multi-Needle Decay Fitter
==================================================
Purpose: Extracts empirical fertility (f) and decay rate (d) from v2 calibration JSONs.
         Uses the 5-needle measurements at varying ages to fit the paper's model:
         P(Recall) = f * (1 - age * d)

Usage:
    python3 fit_decay.py --file test/benchmarks/live_calibration_G2.5-FLASH_100000_v2.json
"""

import json
import argparse
import os
import numpy as np
from scipy.optimize import curve_fit

def linear_decay_model(age, f, d):
    """The paper's fundamental model."""
    # Ensure recall doesn't go below 0
    return np.maximum(0, f * (1 - age * d))

def analyze_calibration(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    ages = []
    recalls = []
    
    # Extract age vs recall binary from the 5-needle results
    for run in data:
        # Support new v3 format
        if "scores" in run:
            is_success = run.get("success", False)
            if is_success:
                for needle in run["scores"]:
                    ages.append(needle.get("age") or needle.get("age_turns"))
                    recalls.append(1.0 if needle.get("recalled") else 0.0)
        # Support legacy v2 format
        elif run.get("status") == 200 and "needle_results" in run:
            for needle in run["needle_results"]:
                ages.append(needle["age_turns"])
                recalls.append(1.0 if needle["recalled"] else 0.0)

    if len(ages) < 10:
        print(f"Warning: Only {len(ages)} data points found. Need more samples for a fit.")
        return

    # Convert to numpy arrays
    x = np.array(ages)
    y = np.array(recalls)

    # Initial guesses: f=0.98, d=1e-7 (tiny decay)
    try:
        popt, pcov = curve_fit(linear_decay_model, x, y, p0=[0.98, 1e-7], bounds=([0, 0], [1, 1e-3]))
        f_est, d_est = popt
        perr = np.sqrt(np.diag(pcov))
        
        print("\n" + "="*50)
        print(f"ANALYSIS RESULTS: {os.path.basename(file_path)}")
        print("="*50)
        print(f"Total Observations: {len(ages)}")
        print(f"Empirical Fidelity (f):  {f_est:.4f}  (±{perr[0]:.4f})")
        print(f"Empirical Decay (d):     {d_est:.4e} (±{perr[1]:.4e})")
        print("-" * 50)
        
        # Calculate theoretical cliff (where P=0.5)
        if d_est > 0:
            cliff_50 = (1 - 0.5/f_est) / d_est
            print(f"Predicted 50% Recall Cliff: {cliff_50:,.0f} turns")
        else:
            print("Predicted 50% Recall Cliff: Infinite (zero decay measured)")
        print("="*50 + "\n")
        
        return f_est, d_est
    except Exception as e:
        print(f"Fit failed: {e}")
        # Fallback: weighted average
        print(f"Average Recall: {np.mean(y):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Needle Decay Fitter")
    parser.add_argument("--file", type=str, required=True, help="Path to v2 calibration JSON")
    args = parser.parse_args()
    analyze_calibration(args.file)
