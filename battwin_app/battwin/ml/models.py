"""The four ML models (maximum) and their small tuning grids."""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

MODELS = ["Gaussian Process", "Extra Trees", "Hist. Gradient Boosting", "Bayesian Ridge"]

GRIDS = {
    "Gaussian Process": {"m__alpha": [1e-6, 1e-3]},
    "Extra Trees": {"m__min_samples_leaf": [2, 8], "m__max_features": [0.5, 1.0]},
    "Hist. Gradient Boosting": {"m__learning_rate": [0.05, 0.15], "m__max_leaf_nodes": [8, 24]},
    "Bayesian Ridge": {"m__alpha_1": [1e-6, 1e-3]},
}
GP_MAX_TRAIN = 700


def make(name: str, seed: int = 0) -> Pipeline:
    if name == "Gaussian Process":
        k = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.1)
        est = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=seed)
    elif name == "Extra Trees":
        est = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    elif name == "Hist. Gradient Boosting":
        est = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
    elif name == "Bayesian Ridge":
        est = BayesianRidge()
    else:
        raise ValueError(f"unknown model {name}; allowed: {MODELS}")
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("sc", StandardScaler()), ("m", est)])


def predict_std(model: Pipeline, X):
    """Predictive std where the model provides one (GP, Bayesian ridge); else None."""
    est = model.named_steps["m"]
    if isinstance(est, (GaussianProcessRegressor, BayesianRidge)):
        Z = model[:-1].transform(X)
        return est.predict(Z, return_std=True)[1]
    return None
