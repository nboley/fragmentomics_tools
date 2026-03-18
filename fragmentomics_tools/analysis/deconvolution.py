import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def fit_celltype_weights_l2(y_df, X_df, y_col="expression", nonnegative=False, ridge=0.0):
    """
    Solve min_w ||y - X w||_2^2 (optionally ridge-regularized and/or nonnegative).

    y_df: DataFrame indexed by gene_id with column y_col
    X_df: DataFrame indexed by gene_id with columns = cell types
    nonnegative: if True, constrain weights >= 0 (uses scipy)
    ridge: L2 regularization strength (0 = ordinary least squares)
    """
    # Align on shared genes
    genes = y_df.index.intersection(X_df.index)
    y = y_df.loc[genes, y_col].astype(float).values
    X = X_df.loc[genes].astype(float).values

    # Drop rows with any NaNs
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y = y[mask]
    X = X[mask]

    if nonnegative:
        from scipy.optimize import nnls
        # nnls doesn't support ridge directly; if you want ridge+nnls, ask and I'll add it
        w, _ = nnls(X, y)
    else:
        # Ordinary least squares or ridge
        if ridge > 0:
            XtX = X.T @ X
            Xty = X.T @ y
            w = np.linalg.solve(XtX + ridge * np.eye(X.shape[1]), Xty)
        else:
            w, *_ = np.linalg.lstsq(X, y, rcond=None)

    weights = pd.Series(w, index=X_df.columns, name="weight")

    # Predictions and fit quality
    y_hat = X @ w
    residual = y - y_hat
    metrics = {
        "n_genes_used": int(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - (np.sum(residual**2) / np.sum((y - y.mean())**2))) if len(y) > 1 else np.nan,
    }

    return weights, metrics


def rank_correlation_y_vs_X(y_df, X_df, y_col="expression"):
    """
    Compute Spearman rank correlation between y and each column of X.

    y_df: DataFrame indexed by gene_id with column y_col
    X_df: DataFrame indexed by gene_id with columns = cell types
    """
    # Align on shared genes
    genes = y_df.index.intersection(X_df.index)
    y = y_df.loc[genes, y_col]

    results = []

    for col in X_df.columns:
        x = X_df.loc[genes, col]

        # Drop genes where either is NaN
        mask = y.notna() & x.notna()
        rho, pval = spearmanr(y[mask], x[mask])

        results.append({
            "cell_type": col,
            "spearman_r": rho,
            "p_value": pval,
            "n_genes": int(mask.sum())
        })

    return (
        pd.DataFrame(results)
        .sort_values("spearman_r", ascending=False)
        .reset_index(drop=True)
    )
