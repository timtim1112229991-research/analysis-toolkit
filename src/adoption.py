"""Models relating perceived attributes to adoption disposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


def _design(frame: pd.DataFrame, outcome: str, predictors: list[str], arm_column: str | None):
    columns = predictors + ([arm_column] if arm_column else [])
    data = frame[[outcome] + columns].dropna().copy()
    if arm_column:
        arms = sorted(data[arm_column].unique())
        data[arm_column] = (data[arm_column] == arms[0]).astype(int)
        data = data.rename(columns={arm_column: f"arm_is_{arms[0]}"})
    y = data[outcome]
    x = data.drop(columns=[outcome])
    return y, x


def ordinal_model(
    frame: pd.DataFrame, outcome: str, predictors: list[str], arm_column: str | None = "arm"
) -> pd.DataFrame:
    """Proportional odds model, appropriate to an ordinal outcome."""
    y, x = _design(frame, outcome, predictors, arm_column)
    model = OrderedModel(y.astype(int), x, distr="logit").fit(method="bfgs", disp=False)
    table = pd.DataFrame(
        {
            "coefficient": model.params,
            "standard_error": model.bse,
            "z": model.tvalues,
            "p_value": model.pvalues,
        }
    )
    table["odds_ratio"] = np.exp(table["coefficient"])
    conf = model.conf_int()
    table["ci_lower"] = conf[0]
    table["ci_upper"] = conf[1]
    table.attrs["log_likelihood"] = float(model.llf)
    table.attrs["observations"] = int(model.nobs)
    return table


def linear_model(
    frame: pd.DataFrame, outcome: str, predictors: list[str], arm_column: str | None = "arm"
) -> pd.DataFrame:
    """Least squares companion, reported for comparability with earlier work."""
    y, x = _design(frame, outcome, predictors, arm_column)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    table = pd.DataFrame(
        {
            "coefficient": model.params,
            "standard_error": model.bse,
            "t": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": model.conf_int()[0],
            "ci_upper": model.conf_int()[1],
        }
    )
    table.attrs["r_squared"] = float(model.rsquared)
    table.attrs["observations"] = int(model.nobs)
    return table


def dimension_correlations(frame: pd.DataFrame, columns: list[str], method: str = "spearman") -> pd.DataFrame:
    return frame[columns].corr(method=method).round(3)
