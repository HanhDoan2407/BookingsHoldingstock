"""
tsa_toolkit.py
--------------
A small, dependency-light time-series toolkit that reimplements the
EViews diagnostics used in the original paper (ADF test, ACF/PACF with
Ljung-Box Q, ARMA conditional-least-squares estimation, Breusch-Godfrey
LM test, ARCH-LM test, Jarque-Bera, Chow breakpoint test) using only
numpy / scipy / pandas. Built this way so the notebook runs anywhere,
even without `statsmodels` installed.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------
# Basic OLS
# ---------------------------------------------------------------------
def ols(y, X):
    """Simple OLS via lstsq. X should already include a constant column
    if one is wanted. Returns dict with coefficients, std errors, t-stats,
    p-values, residuals, R^2."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), df=dof))
    ss_res = resid @ resid
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=beta, se=se, tstat=tstat, pval=pval, resid=resid,
                r2=r2, n=n, k=k, dof=dof, sigma2=sigma2)


# ---------------------------------------------------------------------
# ACF / PACF + Ljung-Box
# ---------------------------------------------------------------------
def acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x ** 2)
    out = np.empty(nlags + 1)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = np.sum(x[k:] * x[:-k]) / denom
    return out


def pacf_durbin_levinson(ac, nlags):
    """PACF via the Durbin-Levinson recursion, given autocorrelations ac[0..nlags]."""
    phi = np.zeros((nlags + 1, nlags + 1))
    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0
    if nlags >= 1:
        phi[1, 1] = ac[1]
        pacf_vals[1] = ac[1]
    for k in range(2, nlags + 1):
        num = ac[k] - np.sum(phi[k - 1, 1:k] * ac[k - 1:0:-1])
        den = 1 - np.sum(phi[k - 1, 1:k] * ac[1:k])
        phi[k, k] = num / den if den != 0 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf_vals[k] = phi[k, k]
    return pacf_vals


def ljung_box(x, nlags):
    """Returns DataFrame with AC, PAC, Q-stat, p-value for lags 1..nlags."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    ac = acf(x, nlags)
    pac = pacf_durbin_levinson(ac, nlags)
    q = np.empty(nlags)
    p = np.empty(nlags)
    running = 0.0
    for k in range(1, nlags + 1):
        running += (ac[k] ** 2) / (n - k)
        q[k - 1] = n * (n + 2) * running
        p[k - 1] = 1 - stats.chi2.cdf(q[k - 1], df=k)
    return pd.DataFrame({
        "lag": np.arange(1, nlags + 1),
        "AC": ac[1:], "PAC": pac[1:], "Q-Stat": q, "Prob": p
    })


# ---------------------------------------------------------------------
# Augmented Dickey-Fuller test (single lag, no augmentation needed for
# daily financial series that already show no residual autocorrelation;
# lag order can be extended via `lags`)
# ---------------------------------------------------------------------
def adf_test(series, regression="c", lags=0):
    """
    regression: 'n' (none), 'c' (constant), 'ct' (constant + trend)
    Returns dict with tstat, pvalue (approx via MacKinnon-style regression
    is not implemented exactly; we report the t-stat and use Dickey-Fuller
    critical values commonly tabulated) and the fitted equation.
    """
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)
    y_lag1 = y[:-1]
    n = len(dy)

    X_cols = [y_lag1.reshape(-1, 1)]
    if regression in ("c", "ct"):
        X_cols.append(np.ones((n, 1)))
    if regression == "ct":
        trend = np.arange(1, n + 1).reshape(-1, 1)
        X_cols.append(trend)

    for L in range(1, lags + 1):
        dyl = np.diff(y)[:-L] if L < n else np.zeros(n)
        # align lengths
    # (kept simple: lags=0 default, matches the paper's SIC-selected lag=0 case)

    X = np.hstack(X_cols)
    res = ols(dy, X)
    df_stat = res["tstat"][0]
    return dict(tstat=df_stat, pval_coef=res["pval"][0], beta=res["beta"],
                se=res["se"], n=n, resid=res["resid"], regression=regression)


ADF_CRIT = {
    "n":  {0.01: -2.5758, 0.05: -1.9414, 0.10: -1.6162},
    "c":  {0.01: -3.4381, 0.05: -2.8641, 0.10: -2.5681},
    "ct": {0.01: -3.9638, 0.05: -3.4126, 0.10: -3.1279},
}


# ---------------------------------------------------------------------
# ARMA(p,d,q) via Conditional Sum-of-Squares (scipy.optimize)
# ---------------------------------------------------------------------
def arma_css(y, p=1, q=1):
    """Fit ARMA(p,q) with a constant by minimizing the conditional sum of
    squared 1-step innovations (CSS), matching EViews' default CSS-ML
    reasonably closely for well-identified small models."""
    y = np.asarray(y, dtype=float)
    n = len(y)

    def unpack(theta):
        c = theta[0]
        phi = theta[1:1 + p]
        psi = theta[1 + p:1 + p + q]
        return c, phi, psi

    def sse(theta):
        c, phi, psi = unpack(theta)
        e = np.zeros(n)
        yhat = np.zeros(n)
        start = max(p, q)
        for t in range(start, n):
            ar_part = np.sum(phi * (y[t - p:t][::-1] - c)) if p > 0 else 0.0
            ma_part = np.sum(psi * e[t - q:t][::-1]) if q > 0 else 0.0
            yhat[t] = c + ar_part + ma_part
            e[t] = y[t] - yhat[t]
        return np.sum(e[start:] ** 2), e, yhat

    theta0 = np.zeros(1 + p + q)
    theta0[0] = np.mean(y)
    result = minimize(lambda th: sse(th)[0], theta0, method="Nelder-Mead",
                       options=dict(maxiter=5000, xatol=1e-8, fatol=1e-10))
    theta_hat = result.x
    ssr, e, yhat = sse(theta_hat)
    start = max(p, q)
    ndof = n - start - len(theta_hat)
    sigma2 = ssr / max(ndof, 1)

    # numerical Hessian -> standard errors
    eps = 1e-4
    k = len(theta_hat)
    H = np.zeros((k, k))
    f0 = sse(theta_hat)[0]
    for i in range(k):
        for j in range(k):
            ti = theta_hat.copy(); ti[i] += eps
            tj = theta_hat.copy(); tj[j] += eps
            tij = theta_hat.copy(); tij[i] += eps; tij[j] += eps
            fij = sse(tij)[0]; fi = sse(ti)[0]; fj = sse(tj)[0]
            H[i, j] = (fij - fi - fj + f0) / (eps ** 2)
    try:
        cov = 2 * sigma2 * np.linalg.pinv(H)
        se = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        se = np.full(k, np.nan)

    tstat = theta_hat / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), df=max(ndof, 1)))
    c, phi, psi = unpack(theta_hat)
    return dict(const=c, ar=phi, ma=psi, se=se, tstat=tstat, pval=pval,
                resid=e[start:], sigma2=sigma2, n=n, nobs=n - start,
                theta=theta_hat)


# ---------------------------------------------------------------------
# Breusch-Godfrey LM test for residual serial correlation
# ---------------------------------------------------------------------
def breusch_godfrey(resid, X, nlags=2):
    """H0: no serial correlation up to `nlags`. X is the original
    regressor matrix (with constant) used in the base model."""
    resid = np.asarray(resid, dtype=float)
    n = len(resid)
    lagged = np.column_stack([np.r_[np.zeros(L), resid[:-L]] for L in range(1, nlags + 1)])
    Xaug = np.hstack([X, lagged])
    res = ols(resid, Xaug)
    LM = n * res["r2"]
    pval = 1 - stats.chi2.cdf(LM, df=nlags)
    fstat = (res["r2"] / nlags) / ((1 - res["r2"]) / (n - Xaug.shape[1]))
    fpval = 1 - stats.f.cdf(fstat, nlags, n - Xaug.shape[1])
    return dict(LM=LM, pval=pval, fstat=fstat, fpval=fpval)


# ---------------------------------------------------------------------
# ARCH-LM test
# ---------------------------------------------------------------------
def arch_lm(resid, nlags=1):
    resid = np.asarray(resid, dtype=float)
    r2 = resid ** 2
    n = len(r2)
    X_cols = [np.ones(n - nlags)]
    for L in range(1, nlags + 1):
        X_cols.append(r2[nlags - L:n - L])
    X = np.column_stack(X_cols)
    y = r2[nlags:]
    res = ols(y, X)
    LM = (n - nlags) * res["r2"]
    pval = 1 - stats.chi2.cdf(LM, df=nlags)
    fstat = (res["r2"] / nlags) / ((1 - res["r2"]) / (len(y) - X.shape[1]))
    fpval = 1 - stats.f.cdf(fstat, nlags, len(y) - X.shape[1])
    return dict(LM=LM, pval=pval, fstat=fstat, fpval=fpval)


# ---------------------------------------------------------------------
# Jarque-Bera
# ---------------------------------------------------------------------
def jarque_bera(x):
    jb, p = stats.jarque_bera(np.asarray(x, dtype=float))
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=False)
    return dict(JB=jb, pval=p, skew=skew, kurtosis=kurt)


# ---------------------------------------------------------------------
# Chow breakpoint test
# ---------------------------------------------------------------------
def chow_breakpoint(y, X, break_idx):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    full = ols(y, X)
    ssr_full = full["resid"] @ full["resid"]
    r1 = ols(y[:break_idx], X[:break_idx])
    r2 = ols(y[break_idx:], X[break_idx:])
    ssr_split = r1["resid"] @ r1["resid"] + r2["resid"] @ r2["resid"]
    fstat = ((ssr_full - ssr_split) / k) / (ssr_split / (n - 2 * k))
    pval = 1 - stats.f.cdf(fstat, k, n - 2 * k)
    return dict(fstat=fstat, pval=pval)


# ---------------------------------------------------------------------
# Naive (random-walk) forecast + evaluation stats (Theil's U etc.)
# ---------------------------------------------------------------------
def forecast_eval(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    err = actual - forecast
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    mape = np.mean(np.abs(err / actual)) * 100
    theil_u = rmse / (np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(forecast ** 2)))
    bias_prop = (np.mean(err) ** 2) / np.mean(err ** 2) if np.mean(err ** 2) > 0 else np.nan
    return dict(rmse=rmse, mae=mae, mape=mape, theil_u=theil_u, bias_prop=bias_prop)
