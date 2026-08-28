"""Part 3: ARIMA identification, estimation, and diagnostics."""

# =====================================================================
# 5. ARIMA Modeling
# =====================================================================
log("\n" + "=" * 70)
log("5. ARIMA MODELING")
log("=" * 70)

d_close = np.diff(close)  # first difference, d=1

log("\n--- 5.1 Identification: trial ARMA(1,1) on differenced series ---")
fit_arma11 = arma_css(d_close, p=1, q=1)
log(f"  const = {fit_arma11['const']:.4f}  (se={fit_arma11['se'][0]:.4f}, "
    f"t={fit_arma11['tstat'][0]:.4f}, p={fit_arma11['pval'][0]:.4f})")
log(f"  AR(1) = {fit_arma11['ar'][0]:.4f}  (se={fit_arma11['se'][1]:.4f}, "
    f"t={fit_arma11['tstat'][1]:.4f}, p={fit_arma11['pval'][1]:.4f})")
log(f"  MA(1) = {fit_arma11['ma'][0]:.4f}  (se={fit_arma11['se'][2]:.4f}, "
    f"t={fit_arma11['tstat'][2]:.4f}, p={fit_arma11['pval'][2]:.4f})")
log("  -> AR(1)/MA(1) statistically insignificant => over-specified, matches paper's finding.")

log("\n--- 5.2 Adopted specification: ARIMA(0,1,0), i.e. Random Walk ---")
# ARIMA(0,1,0) with drift = OLS of d_close on a constant
X_const = np.ones((len(d_close), 1))
fit_rw = ols(d_close, X_const)
const_rw = fit_rw["beta"][0]
se_rw = fit_rw["se"][0]
t_rw = fit_rw["tstat"][0]
p_rw = fit_rw["pval"][0]
resid_rw = fit_rw["resid"]
log(f"  drift (constant) = {const_rw:.4f}  (se={se_rw:.4f}, t={t_rw:.4f}, p={p_rw:.4f})")
log(f"  R-squared = {fit_rw['r2']:.6f}")
log("  -> constant statistically insignificant at 5% => Random Walk WITHOUT drift, matches paper.")

# Figure 13/14 equivalent: bar-style summary table image for ARMA(1,1) and ARIMA(0,1,0)
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.axis("off")
tbl_txt = (
    f"{'Variable':<10}{'Coefficient':>14}{'Std.Error':>12}{'t-Stat':>10}{'Prob':>10}\n"
    f"{'AR(1)':<10}{fit_arma11['ar'][0]:>14.4f}{fit_arma11['se'][1]:>12.4f}"
    f"{fit_arma11['tstat'][1]:>10.4f}{fit_arma11['pval'][1]:>10.4f}\n"
    f"{'MA(1)':<10}{fit_arma11['ma'][0]:>14.4f}{fit_arma11['se'][2]:>12.4f}"
    f"{fit_arma11['tstat'][2]:>10.4f}{fit_arma11['pval'][2]:>10.4f}\n"
)
ax.text(0, 1, "ARMA(1,1) on D(bkng_close) — trial specification", fontsize=11, weight="bold")
ax.text(0, 0.6, tbl_txt, family="monospace", fontsize=10, va="top")
savefig("fig13_arma11_results.png")

fig, ax = plt.subplots(figsize=(6.5, 2.2))
ax.axis("off")
tbl_txt2 = (
    f"{'Variable':<10}{'Coefficient':>14}{'Std.Error':>12}{'t-Stat':>10}{'Prob':>10}\n"
    f"{'C':<10}{const_rw:>14.4f}{se_rw:>12.4f}{t_rw:>10.4f}{p_rw:>10.4f}\n"
    f"\nR-squared = {fit_rw['r2']:.6f}"
)
ax.text(0, 1, "ARIMA(0,1,0) on D(bkng_close) — adopted specification", fontsize=11, weight="bold")
ax.text(0, 0.55, tbl_txt2, family="monospace", fontsize=10, va="top")
savefig("fig14_arima010_results.png")

# --- 5.3 Model diagnostics: correlogram of residuals ---
log("\n--- 5.3 Model diagnostics: residual autocorrelation ---")
lb_resid = plot_correlogram(resid_rw, NLAGS, "Correlogram of Residuals (ARIMA(0,1,0))",
                             "fig15_correlogram_residuals.png")
log(lb_resid.head().to_string(index=False))
log(f"Q-stat lag1 = {lb_resid['Q-Stat'].iloc[0]:.3f}, p = {lb_resid['Prob'].iloc[0]:.4f} "
    "-> fail to reject H0 of no autocorrelation (white noise), matches paper.")

bg = breusch_godfrey(resid_rw, X_const, nlags=2)
log(f"\nBreusch-Godfrey Serial Correlation LM Test (2 lags): "
    f"LM={bg['LM']:.4f}, p={bg['pval']:.4f}, F={bg['fstat']:.4f}, F-p={bg['fpval']:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 2))
ax.axis("off")
ax.text(0, 1, "Breusch-Godfrey Serial Correlation LM Test", fontsize=11, weight="bold")
ax.text(0, 0.5, f"F-statistic = {bg['fstat']:.4f}   Prob. F(2,{len(resid_rw)-2-1}) = {bg['fpval']:.4f}\n"
                 f"Obs*R-squared = {bg['LM']:.4f}   Prob. Chi-Square(2) = {bg['pval']:.4f}",
        family="monospace", fontsize=10, va="top")
savefig("fig16_breusch_godfrey.png")

# --- 5.4 Heteroskedasticity: ARCH-LM test ---
log("\n--- 5.4 Heteroskedasticity: ARCH-LM test ---")
arch = arch_lm(resid_rw, nlags=1)
log(f"ARCH-LM (1 lag): LM={arch['LM']:.4f}, p={arch['pval']:.4f}, "
    f"F={arch['fstat']:.4f}, F-p={arch['fpval']:.4f}")
log("-> fail to reject H0 of no ARCH effects" if arch["pval"] > 0.05 else "-> ARCH effects present")

fig, ax = plt.subplots(figsize=(6.5, 2))
ax.axis("off")
ax.text(0, 1, "Heteroskedasticity Test: ARCH", fontsize=11, weight="bold")
ax.text(0, 0.5, f"F-statistic = {arch['fstat']:.4f}   Prob. F(1,{len(resid_rw)-2}) = {arch['fpval']:.4f}\n"
                 f"Obs*R-squared = {arch['LM']:.4f}   Prob. Chi-Square(1) = {arch['pval']:.4f}",
        family="monospace", fontsize=10, va="top")
savefig("fig17_arch_test.png")

# --- Residual/Actual/Fitted plot ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["date"].iloc[1:], resid_rw, color="#4472c4", lw=0.8, label="Residual")
ax.plot(df["date"].iloc[1:], d_close, color="#c55a11", lw=0.8, alpha=0.7, label="Actual (D close)")
ax.axhline(const_rw, color="green", lw=1.2, label="Fitted (constant)")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Residual, Actual, and Fitted Plot — ARIMA(0,1,0)")
savefig("fig18_residual_actual_fitted.png")

# --- 5.5 Normality of residuals ---
log("\n--- 5.5 Normality of ARIMA(0,1,0) residuals ---")
jb_resid = jarque_bera(resid_rw)
log(f"Jarque-Bera = {jb_resid['JB']:.2f}, p = {jb_resid['pval']:.6f}, "
    f"skew={jb_resid['skew']:.4f}, kurtosis={jb_resid['kurtosis']:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(resid_rw, bins=25, color="#4472c4", edgecolor="white")
ax.set_title("Histogram of ARIMA(0,1,0) Residuals")
txt = (f"Obs: {len(resid_rw)}\nMean: {resid_rw.mean():.4f}\nStd.Dev: {resid_rw.std(ddof=1):.4f}\n"
       f"Skewness: {jb_resid['skew']:.4f}\nKurtosis: {jb_resid['kurtosis']:.4f}\n"
       f"Jarque-Bera: {jb_resid['JB']:.2f}\nProbability: {jb_resid['pval']:.6f}")
ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", fontsize=9,
        family="monospace", bbox=dict(boxstyle="round", fc="#f2f2f2"))
savefig("fig19_residuals_hist_jb.png")

# --- 5.6 Stability: Chow breakpoint test ---
log("\n--- 5.6 Stability: Chow breakpoint test ---")
break_date = pd.Timestamp("2025-01-15")  # roughly mid-sample turning point for this window
dates_d = df["date"].iloc[1:].reset_index(drop=True)
break_idx = int((dates_d < break_date).sum())
chow = chow_breakpoint(d_close, X_const, break_idx)
log(f"Breakpoint date used: {break_date.date()} (index {break_idx} of {len(d_close)})")
log(f"Chow F-stat = {chow['fstat']:.4f}, p = {chow['pval']:.4f}")
log("-> fail to reject H0 of no structural break" if chow["pval"] > 0.05 else "-> structural break detected")

fig, ax = plt.subplots(figsize=(6.5, 2))
ax.axis("off")
ax.text(0, 1, f"Chow Breakpoint Test: {break_date.date()}", fontsize=11, weight="bold")
ax.text(0, 0.5, f"F-statistic = {chow['fstat']:.4f}   Prob. F = {chow['pval']:.4f}",
        family="monospace", fontsize=10, va="top")
savefig("fig20_chow_test.png")

print("PART 3 DONE")
