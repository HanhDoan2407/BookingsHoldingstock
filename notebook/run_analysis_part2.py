"""Part 2: Autocorrelation analysis + ADF unit-root tests.
Appended logic to run_analysis.py (imported after part 1 runs)."""

# =====================================================================
# 3. Autocorrelation analysis
# =====================================================================
log("\n" + "=" * 70)
log("3. AUTOCORRELATION ANALYSIS")
log("=" * 70)

NLAGS = 20


def plot_correlogram(x, nlags, title, fname):
    lb = ljung_box(x, nlags)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(lb["lag"], lb["AC"], color="#4472c4")
    axes[0].axhline(0, color="black", lw=0.6)
    ci = 1.96 / np.sqrt(len(x))
    axes[0].axhline(ci, color="red", ls="--", lw=0.8)
    axes[0].axhline(-ci, color="red", ls="--", lw=0.8)
    axes[0].set_title("Autocorrelation (AC)")
    axes[0].set_xlabel("Lag")

    axes[1].bar(lb["lag"], lb["PAC"], color="#c55a11")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].axhline(ci, color="red", ls="--", lw=0.8)
    axes[1].axhline(-ci, color="red", ls="--", lw=0.8)
    axes[1].set_title("Partial Autocorrelation (PAC)")
    axes[1].set_xlabel("Lag")
    fig.suptitle(title)
    savefig(fname)
    return lb


lb_close = plot_correlogram(close, NLAGS, "Correlogram of bkng_close", "fig08_correlogram_close.png")
log("Correlogram of bkng_close (first 5 lags):")
log(lb_close.head().to_string(index=False))
log(f"Q-stat lag1 = {lb_close['Q-Stat'].iloc[0]:.2f}, p = {lb_close['Prob'].iloc[0]:.4f}")

lb_logret = plot_correlogram(logret, NLAGS, "Correlogram of bkng_logret", "fig09_correlogram_logret.png")
log("\nCorrelogram of bkng_logret (first 5 lags):")
log(lb_logret.head().to_string(index=False))
log(f"Q-stat lag1 = {lb_logret['Q-Stat'].iloc[0]:.2f}, p = {lb_logret['Prob'].iloc[0]:.4f}")

sq_logret = logret ** 2
lb_sq = plot_correlogram(sq_logret, NLAGS, "Correlogram of squared log returns", "fig12_correlogram_sq_logret.png")
log("\nCorrelogram of squared log returns (first 5 lags):")
log(lb_sq.head().to_string(index=False))

# =====================================================================
# 4. Augmented Dickey-Fuller tests
# =====================================================================
log("\n" + "=" * 70)
log("4. AUGMENTED DICKEY-FULLER (ADF) UNIT ROOT TESTS")
log("=" * 70)


def report_adf(series, name):
    log(f"\n--- {name} ---")
    for reg in ["n", "c", "ct"]:
        res = adf_test(series, regression=reg)
        crit = ADF_CRIT[reg]
        log(f"  [{reg:>2}] ADF t-stat = {res['tstat']:.4f}   "
            f"crit(1%,5%,10%) = {crit[0.01]:.4f}, {crit[0.05]:.4f}, {crit[0.10]:.4f}   "
            f"{'REJECT H0 (stationary)' if res['tstat'] < crit[0.05] else 'FAIL TO REJECT H0 (unit root)'}")
    return res


report_adf(close, "BKNG_CLOSE")
report_adf(logret, "BKNG_LOGRET")

print("PART 2 DONE")
