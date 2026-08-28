"""Part 4: Forecasting (naive random-walk forecast) + wrap-up."""

# =====================================================================
# 5.7 Forecasting
# =====================================================================
log("\n" + "=" * 70)
log("5.7 FORECASTING")
log("=" * 70)

HORIZON = 15  # trading days, mirroring the paper's 2-week horizon
train_close = close[:-HORIZON]
test_close = close[-HORIZON:]
test_dates = df["date"].iloc[-HORIZON:].reset_index(drop=True)

# naive / random-walk forecast: all future values = last observed level
last_level = train_close[-1]
point_forecast = np.full(HORIZON, last_level)

# growing confidence interval: variance of RW forecast at horizon h is h * sigma^2
sigma_e = resid_rw.std(ddof=1)
h = np.arange(1, HORIZON + 1)
se_h = sigma_e * np.sqrt(h)
ci_upper = point_forecast + 1.96 * se_h
ci_lower = point_forecast - 1.96 * se_h

fe = forecast_eval(test_close, point_forecast)
log(f"Forecast horizon: {HORIZON} trading days "
    f"({test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()})")
log(f"RMSE={fe['rmse']:.4f}  MAE={fe['mae']:.4f}  MAPE={fe['mape']:.4f}%  "
    f"Theil U={fe['theil_u']:.6f}  Bias Proportion={fe['bias_prop']:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 3))
ax.axis("off")
ax.text(0, 1, "Forecast Evaluation Statistics", fontsize=11, weight="bold")
ax.text(0, 0.7,
        f"Root Mean Squared Error   {fe['rmse']:.4f}\n"
        f"Mean Absolute Error       {fe['mae']:.4f}\n"
        f"Mean Abs. Percent Error   {fe['mape']:.4f}\n"
        f"Theil Inequality Coef.    {fe['theil_u']:.6f}\n"
        f"Bias Proportion           {fe['bias_prop']:.4f}",
        family="monospace", fontsize=10, va="top")
savefig("fig21_forecast_eval_stats.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(df["date"], close, color="#1f5fa8", lw=1.1, label="BKNG close price (actual)")
ax.plot(test_dates, point_forecast, color="#1f5fa8", lw=1.6, ls="-", label="Forecast (naive RW)")
ax.fill_between(test_dates, ci_lower, ci_upper, color="red", alpha=0.15, label="95% CI")
ax.plot(test_dates, ci_upper, color="red", lw=0.8, ls="--")
ax.plot(test_dates, ci_lower, color="red", lw=0.8, ls="--")
ax.axvspan(test_dates.iloc[0], test_dates.iloc[-1], color="grey", alpha=0.08)
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Actual vs. Model Forecast with 95% Confidence Intervals")
savefig("fig22_actual_vs_forecast.png")

log("\n" + "=" * 70)
log("6. CONCLUSION")
log("=" * 70)
log(f"""
Over {df['date'].min().date()} to {df['date'].max().date()} ({len(close)} obs, real BKNG
data, rescaled x25 to the pre-split nominal level for comparability with
the original paper), BKNG's closing price series is non-stationary
(ADF fails to reject a unit root at all three specifications), while
first-differenced log returns are stationary and close to white noise.
An ARMA(1,1) trial on the differenced series was over-specified (both
AR(1) and MA(1) insignificant, p > 0.9), so the parsimonious ARIMA(0,1,0)
"random walk" model was adopted, matching the original EViews-based study.
The constant/drift term is statistically insignificant (p = {p_rw:.3f}), so the
series behaves as a random walk WITHOUT drift. Residual diagnostics
(Ljung-Box, Breusch-Godfrey) show no remaining autocorrelation; the
ARCH-LM test shows no volatility clustering; the Chow breakpoint test
shows no structural break; and residuals are leptokurtic but not normal
(a standard stylized fact of financial returns) -- all consistent with
the original paper's findings.
""")

with open(os.path.join(HERE, "..", "analysis_log.txt"), "w") as f:
    f.write("\n".join(LOG_LINES))

log(f"\nLog written to analysis_log.txt. {len(os.listdir(FIG_DIR))} figures saved to /figures.")
print("PART 4 DONE — FULL PIPELINE COMPLETE")
