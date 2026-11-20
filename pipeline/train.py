import argparse
import hashlib
import json
import time
import warnings
from datetime import date
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from features import LAG, make_baseline, make_pipeline, prepare

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
TOL = 0.005


def rmsle(y_true, y_pred):
    p = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.asarray(y_true) - p) ** 2)))


def mae_log(y_true, y_pred):
    return float(mean_absolute_error(y_true, np.clip(y_pred, 0, None)))


def r2_log(y_true, y_pred):
    return float(r2_score(y_true, np.clip(y_pred, 0, None)))


SCORING = {
    "rmsle": make_scorer(rmsle, greater_is_better=False),
    "mae": make_scorer(mae_log, greater_is_better=False),
    "r2": make_scorer(r2_log),
}


def walk_forward_splits(dates, n_folds=5, val_days=30, gap_days=LAG):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    end, splits = dates.max(), []
    for k in range(n_folds, 0, -1):
        val_end = end - pd.Timedelta(days=(k - 1) * val_days)
        val_start = val_end - pd.Timedelta(days=val_days)
        train_end = val_start - pd.Timedelta(days=gap_days)
        tr = np.where(dates <= train_end)[0]
        va = np.where((dates > val_start) & (dates <= val_end))[0]
        if len(tr) and len(va):
            splits.append((tr, va))
    return splits


LOG = []


def run(name, est, X, y, splits, params="", extra=None, return_train=False):
    t0 = time.time()
    r = cross_validate(
        est,
        X,
        y,
        cv=splits,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=return_train,
        return_estimator=True,
    )
    row = {
        "model": name,
        "params": params,
        "rmsle_mean": -r["test_rmsle"].mean(),
        "rmsle_std": r["test_rmsle"].std(),
        "mae_log": -r["test_mae"].mean(),
        "r2": r["test_r2"].mean(),
        "fit_time_s": r["fit_time"].mean(),
        "date": date.today().isoformat(),
    }
    if extra:
        row.update(extra)
    LOG.append(row)
    print(
        f"{name:45s} RMSLE={row['rmsle_mean']:.4f} ± {row['rmsle_std']:.4f} "
        f"time={row['fit_time_s']:.1f}s ({time.time() - t0:.0f}s total)"
    )
    return row, r


def save_log(path="reports/experiments.csv"):
    pd.DataFrame(LOG).to_csv(path, index=False)


def get_models():
    models = {
        "Ridge (L2)": Ridge(alpha=1.0, random_state=SEED),
        "Lasso (L1)": Lasso(alpha=1e-3, random_state=SEED, max_iter=5000),
        "kNN (k=25)": KNeighborsRegressor(n_neighbors=25, n_jobs=-1),
        "Decision tree (depth=8)": DecisionTreeRegressor(
            max_depth=8, random_state=SEED
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=SEED),
    }
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=-1)
    except ImportError:
        pass
    return models


GRIDS = {
    "Ridge (L2)": {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
    "Random forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [3, 5, 10],
        "model__max_features": [0.5, 0.8, 1.0],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 6, None],
        "model__max_iter": [150, 300],
        "model__min_samples_leaf": [20, 50, 100],
        "model__l2_regularization": [0.0, 1.0],
    },
    "LightGBM": {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
        "model__n_estimators": [150, 300],
        "model__min_child_samples": [20, 50, 100],
    },
}


def plot_regularization(df_ridge, df_lasso, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(df_ridge["alpha"], df_ridge["rmsle"], "o-", label="L2 (Ridge)")
    ax[0].semilogx(df_lasso["alpha"], df_lasso["rmsle"], "s-", label="L1 (Lasso)")
    ax[0].set_xlabel("alpha (сила регуляризации)")
    ax[0].set_ylabel("RMSLE (CV)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].semilogx(df_ridge["alpha"], df_ridge["nnz"], "o-", label="L2 (Ridge)")
    ax[1].semilogx(df_lasso["alpha"], df_lasso["nnz"], "s-", label="L1 (Lasso)")
    ax[1].set_xlabel("alpha (сила регуляризации)")
    ax[1].set_ylabel("число ненулевых коэффициентов")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_depth(depths, tr, va, path):
    plt.figure(figsize=(6, 4))
    plt.plot(depths, tr, "o-", label="обучающие")
    plt.plot(depths, va, "s-", label="проверочные")
    plt.xlabel("глубина дерева")
    plt.ylabel("RMSLE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(df, path):
    d = df.sort_values("rmsle_mean", ascending=False)
    plt.figure(figsize=(9, 0.4 * len(d) + 1.5))
    plt.barh(d["model"], d["rmsle_mean"], xerr=d["rmsle_std"])
    plt.xlabel("RMSLE (walk-forward CV, меньше - лучше)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(a):
    for d in ["reports", "models", "artifacts"]:
        Path(d).mkdir(exist_ok=True)

    X, y = prepare(a.train, tables_dir=a.raw_dir)
    X2, y2 = prepare(a.train, tables_dir=a.raw_dir)
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)
    h = lambda d: hashlib.md5(
        pd.util.hash_pandas_object(d, index=True).values.tobytes()
    ).hexdigest()
    print(f"Повторяемость: OK, hash X={h(X)} ({len(X)} строк, {X.shape[1]} столбцов)")

    if len(X) > a.sample:
        idx = X.sample(n=a.sample, random_state=SEED).index.sort_values()
        Xs, ys = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
    else:
        Xs, ys = X, y
    splits = walk_forward_splits(Xs["date"], a.folds)
    for i, (tr, va) in enumerate(splits, 1):
        print(
            f"fold {i}: train до {Xs['date'].iloc[tr].max().date()} "
            f"(n={len(tr)}) | val {Xs['date'].iloc[va].min().date()} - "
            f"{Xs['date'].iloc[va].max().date()} (n={len(va)})"
        )

    run("Baseline: mean(log1p sales)", DummyRegressor(strategy="mean"), Xs, ys, splits)
    run("Baseline: median (store, family, dow)", make_baseline(), Xs, ys, splits)

    reg = {"Ridge": [], "Lasso": []}
    for alpha in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        row, r = run(
            f"Ridge alpha={alpha}",
            make_pipeline(Ridge(alpha=alpha)),
            Xs,
            ys,
            splits,
            params=f"alpha={alpha}",
        )
        nnz = np.mean(
            [np.count_nonzero(e.named_steps["model"].coef_) for e in r["estimator"]]
        )
        reg["Ridge"].append({"alpha": alpha, "rmsle": row["rmsle_mean"], "nnz": nnz})
    for alpha in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]:
        row, r = run(
            f"Lasso alpha={alpha}",
            make_pipeline(Lasso(alpha=alpha, max_iter=5000)),
            Xs,
            ys,
            splits,
            params=f"alpha={alpha}",
        )
        nnz = np.mean(
            [np.count_nonzero(e.named_steps["model"].coef_) for e in r["estimator"]]
        )
        reg["Lasso"].append({"alpha": alpha, "rmsle": row["rmsle_mean"], "nnz": nnz})
    df_r, df_l = pd.DataFrame(reg["Ridge"]), pd.DataFrame(reg["Lasso"])
    plot_regularization(df_r, df_l, "reports/fig_regularization.png")
    df_r.to_csv("reports/regularization_ridge.csv", index=False)
    df_l.to_csv("reports/regularization_lasso.csv", index=False)

    for nm, df_, cls in [("ridge", df_r, Ridge), ("lasso", df_l, Lasso)]:
        best_alpha = df_.sort_values("rmsle").iloc[0]["alpha"]
        pipe = make_pipeline(cls(alpha=best_alpha)).fit(Xs, ys)
        names = pipe.named_steps["prep"].get_feature_names_out()
        coef = pd.DataFrame({"feature": names, "coef": pipe.named_steps["model"].coef_})
        coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
        coef.to_csv(f"reports/coefficients_{nm}.csv", index=False)
        print(
            f"\nКоэффициенты {nm} (alpha={best_alpha}):\n{coef.to_string(index=False)}\n"
        )

    models = get_models()
    for name in [
        "kNN (k=25)",
        "Decision tree (depth=8)",
        "Random forest",
        "HistGradientBoosting",
        "LightGBM",
    ]:
        if name in models:
            run(name, make_pipeline(models[name]), Xs, ys, splits)
    run("Ridge (L2)", make_pipeline(models["Ridge (L2)"]), Xs, ys, splits, "alpha=1")

    depths = [2, 3, 5, 8, 12, 16, 20, 30]
    tr_s, va_s = [], []
    for dpt in depths:
        _, r = run(
            f"Decision tree depth={dpt}",
            make_pipeline(DecisionTreeRegressor(max_depth=dpt, random_state=SEED)),
            Xs,
            ys,
            splits,
            params=f"max_depth={dpt}",
            return_train=True,
        )
        tr_s.append(-r["train_rmsle"].mean())
        va_s.append(-r["test_rmsle"].mean())
    plot_depth(depths, tr_s, va_s, "reports/fig_tree_depth.png")

    for nm, mdl in [
        ("Ridge", Ridge(alpha=1.0)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=SEED)),
    ]:
        for flag in (False, True):
            run(
                f"{nm}: {'с новыми' if flag else 'без новых'} признаков",
                make_pipeline(clone(mdl), new_features=flag),
                Xs,
                ys,
                splits,
                params=f"new_features={flag}",
            )

    lg = pd.DataFrame(LOG)
    cands = (
        lg[lg["model"].isin(list(GRIDS))]
        .sort_values("rmsle_mean")
        .drop_duplicates("model")
        .head(2)["model"]
        .tolist()
    )
    tuned = {}
    for name in cands:
        search = RandomizedSearchCV(
            make_pipeline(clone(models[name])),
            GRIDS[name],
            n_iter=min(a.n_iter, int(np.prod([len(v) for v in GRIDS[name].values()]))),
            scoring=SCORING["rmsle"],
            cv=splits,
            refit=False,
            random_state=SEED,
            n_jobs=1,
        )
        search.fit(Xs, ys)
        best = {k: (None if v is None else v) for k, v in search.best_params_.items()}
        print(f"\n{name}: лучший набор {best}, RMSLE={-search.best_score_:.4f}")
        pd.DataFrame(search.cv_results_).to_csv(
            f"reports/tuning_{name.split()[0].lower()}.csv", index=False
        )
        est = make_pipeline(clone(models[name])).set_params(**best)
        tuned[f"{name} (tuned)"] = est
        run(
            f"{name} (tuned)", est, Xs, ys, splits, params=json.dumps(best, default=str)
        )

    save_log()
    res = pd.DataFrame(LOG)
    plot_comparison(res, "reports/fig_model_comparison.png")
    pool = res[
        ~res["model"].str.startswith(
            ("Baseline", "Decision tree depth", "Ridge alpha", "Lasso alpha")
        )
        & ~res["model"].str.contains("признаков")
    ]
    best_rmsle = pool["rmsle_mean"].min()
    final_row = (
        pool[pool["rmsle_mean"] <= best_rmsle + TOL].sort_values("fit_time_s").iloc[0]
    )
    final_name = final_row["model"]
    print(f"\nИтоговая модель: {final_name} (RMSLE={final_row['rmsle_mean']:.4f})")

    if final_name in tuned:
        final_est = tuned[final_name]
    else:
        final_est = make_pipeline(clone(models[final_name]))

    final_est.fit(X, y)
    joblib.dump(final_est, "models/model_v1.joblib")
    info = {
        "version": "v1",
        "trained": date.today().isoformat(),
        "data": f"{Path(a.train).name}, {len(X)} строк, "
        f"{X['date'].min().date()}..{X['date'].max().date()}, обучение на 100 %",
        "algorithm": final_name,
        "target": "log1p(sales); прогноз -> expm1",
        "metric": "RMSLE",
        "cv": f"walk-forward, {len(splits)} фолдов по 30 дней, зазор {LAG} дней",
        "cv_rmsle": round(float(final_row["rmsle_mean"]), 4),
        "params": final_row["params"],
    }
    json.dump(info, open("models/model_v1.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    print("\nСводная таблица:\n", res.sort_values("rmsle_mean").to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/raw/train_base.csv")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument(
        "--sample", type=int, default=200_000, help="строк для сравнения моделей"
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=12)
    main(p.parse_args())
