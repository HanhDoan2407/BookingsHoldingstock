"""
run_analysis.py
----------------
Python replication of "Analysis of Booking Holdings Inc. Stock Volatility
and ARIMA Modeling" (originally built in EViews). Produces every figure
from the paper plus a text log of every diagnostic statistic.

Data note: BKNG completed a 25-for-1 forward stock split in April 2026.
Prices below are rescaled (x25) back to the pre-split nominal scale so
they are directly comparable to the original paper's figures ($3,000-
$5,800 range). See README.md for details and for how to pull the FULL
Jan 2024-May 2025 window via yfinance with a live internet connection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tsa_toolkit import (ols, acf, pacf_durbin_levinson, ljung_box, adf_test, ADF_CRIT,
                          arma_css, breusch_godfrey, arch_lm, jarque_bera,
                          chow_breakpoint, forecast_eval)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                      "grid.alpha": 0.3, "figure.facecolor": "white"})

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"[saved figure] {name}")


# =====================================================================
# 1. Data
# =====================================================================
log("=" * 70)
log("1. DATA")
log("=" * 70)

DATA_PATH = os.path.join(HERE, "..", "data", "bkng_close_2024_2025_demo.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["bkng_logret"] = np.log(df["bkng_close"] / df["bkng_close"].shift(1))
df = df.dropna().reset_index(drop=True)  # first obs loses logret

log(f"Sample: {df['date'].min().date()} to {df['date'].max().date()}  "
    f"({len(df)} observations)")
log(df["bkng_close"].describe().to_string())

# =====================================================================
# 2. Elementary statistical analysis
# =====================================================================
log("\n" + "=" * 70)
log("2. ELEMENTARY STATISTICAL ANALYSIS")
log("=" * 70)

# Figure 1: trend of close price
plt.figure(figsize=(8, 4))
plt.plot(df["date"], df["bkng_close"], color="#1f5fa8", lw=1.1)
plt.title("BKNG Closing Price (pre-split-adjusted)")
plt.ylabel("USD")
savefig("fig01_bkng_close_trend.png")

# Figure 2: histogram + descriptive stats of close price
close = df["bkng_close"].values
desc = dict(mean=close.mean(), median=np.median(close), maximum=close.max(),
            minimum=close.min(), std=close.std(ddof=1))
jb_close = jarque_bera(close)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(close, bins=20, color="#4472c4", edgecolor="white")
ax.set_title("Histogram of bkng_close")
txt = (f"Obs: {len(close)}\nMean: {desc['mean']:.2f}\nMedian: {desc['median']:.2f}\n"
       f"Max: {desc['maximum']:.2f}\nMin: {desc['minimum']:.2f}\nStd.Dev: {desc['std']:.2f}\n"
       f"Skewness: {jb_close['skew']:.4f}\nKurtosis: {jb_close['kurtosis']:.4f}\n"
       f"Jarque-Bera: {jb_close['JB']:.2f}\nProbability: {jb_close['pval']:.6f}")
ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", fontsize=9,
        family="monospace", bbox=dict(boxstyle="round", fc="#f2f2f2"))
savefig("fig02_bkng_close_hist.png")

log(f"bkng_close: mean={desc['mean']:.2f}, median={desc['median']:.2f}, "
    f"std={desc['std']:.2f}, skew={jb_close['skew']:.4f}, kurt={jb_close['kurtosis']:.4f}")
log(f"Jarque-Bera (close) = {jb_close['JB']:.2f}, p = {jb_close['pval']:.6f}")

# Figure 3: Q-Q plot of close price
fig = plt.figure(figsize=(5.5, 5))
stats.probplot(close, dist="norm", plot=plt)
plt.title("Q-Q Plot: bkng_close vs Normal")
savefig("fig03_bkng_close_qq.png")

# Figure 4: trend of log returns
logret = df["bkng_logret"].values
plt.figure(figsize=(8, 4))
plt.plot(df["date"], logret, color="#1f5fa8", lw=0.8)
plt.axhline(0, color="black", lw=0.6)
plt.title("BKNG Log Returns")
savefig("fig04_bkng_logret_trend.png")

# Figure 5: histogram of log returns
jb_logret = jarque_bera(logret)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(logret, bins=30, color="#4472c4", edgecolor="white")
ax.set_title("Histogram of bkng_logret")
txt = (f"Obs: {len(logret)}\nMean: {logret.mean():.6f}\nMedian: {np.median(logret):.6f}\n"
       f"Max: {logret.max():.6f}\nMin: {logret.min():.6f}\nStd.Dev: {logret.std(ddof=1):.6f}\n"
       f"Skewness: {jb_logret['skew']:.4f}\nKurtosis: {jb_logret['kurtosis']:.4f}\n"
       f"Jarque-Bera: {jb_logret['JB']:.2f}\nProbability: {jb_logret['pval']:.6f}")
ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", fontsize=9,
        family="monospace", bbox=dict(boxstyle="round", fc="#f2f2f2"))
savefig("fig05_bkng_logret_hist.png")

log(f"bkng_logret: mean={logret.mean():.6f}, std={logret.std(ddof=1):.6f}, "
    f"skew={jb_logret['skew']:.4f}, kurt={jb_logret['kurtosis']:.4f}")
log(f"Jarque-Bera (logret) = {jb_logret['JB']:.2f}, p = {jb_logret['pval']:.6f}")

# Figure 6: Q-Q plot of log returns
fig = plt.figure(figsize=(5.5, 5))
stats.probplot(logret, dist="norm", plot=plt)
plt.title("Q-Q Plot: bkng_logret vs Normal")
savefig("fig06_bkng_logret_qq.png")

# Figure 7: overlay of close price and log returns
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(df["date"], close, color="#c55a11", label="bkng_close")
ax1.set_ylabel("bkng_close", color="#c55a11")
ax2 = ax1.twinx()
ax2.plot(df["date"], logret, color="#1f5fa8", lw=0.7, alpha=0.8, label="bkng_logret")
ax2.set_ylabel("bkng_logret", color="#1f5fa8")
plt.title("Trend of bkng_logret and BKNG close prices")
savefig("fig07_overlay_close_logret.png")

print("PART 1 DONE")
