"""Helpers for the COMP0173 exploratory data analysis lab.

Students don't need to read this file. It does four quiet jobs:

1. checks that the installed libraries match the versions the lab was tested with;
2. hands out data files only after checking their SHA-256 fingerprint;
3. sets the plotting style and provides helpers for the figures;
4. fits and scores the lab's models, so that the notebook can concentrate on the data.
"""
import hashlib
import urllib.request
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold

# Where the data lives. Locally it is the data/ folder; once published, Colab
# downloads each file from LAB_BASE_URL into DATA_DIR on first use.
LAB_BASE_URL = ""  # set to the GitHub raw URL (pinned to a commit) when the lab is published
DATA_DIR = Path("data")

CHECKSUMS = {
    "sustainable_energy.csv": "36ab95fcb4bd22b81f1093f683ddee5bc133810bdfb29ccab7629ddcc96e3ccd",
    "world_bank_countries.csv": "cc9c00fbd740306961a13a663893c4c520b2e4837b1cf49a05beb8d7c04cd123",
    "datasaurus.tsv": "c8c4792c3ed15238ba07dad6ab778adc79d72e3e1324ead4d8c1ae52849780f8",
    "world_bank_regions.csv": "5d1c336d85ce7d778031eddf89ffd29e01078d125f8d607f23f8a50cb5c2f998",
}

# Versions the lab was tested with: Google Colab, October 2026.
TESTED_VERSIONS = {
    "pandas": "2.2.3", "numpy": "2.1.3", "matplotlib": "3.10.0", "seaborn": "0.13.2",
    "scipy": "1.16.3", "scikit-learn": "1.6.1", "statsmodels": "0.15.0",
    "plotly": "5.24.1", "missingno": "0.5.2",
}

# Colour roles. The same colour always means the same thing.
BLUE = "#2a78d6"    # after EDA: the correct version
ORANGE = "#eb6834"  # skipped EDA: the naive version
AQUA = "#1baf7a"    # a third group, used rarely and always labelled
GREY = "#898781"    # context: everything that isn't the focus
RED = "#d03b3b"     # an error flag, always with a text label
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _installed(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _check_versions():
    """Stay silent when everything matches; explain politely when it doesn't."""
    drifted = {p: (_installed(p), t) for p, t in TESTED_VERSIONS.items() if _installed(p) != t}
    if drifted:
        lines = [f"  {p}: you have {have}, the lab was tested with {tested}" for p, (have, tested) in drifted.items()]
        warnings.warn(
            "Some libraries differ from the versions this lab was tested with. "
            "Everything should still run, but a few numbers may differ slightly.\n" + "\n".join(lines),
            stacklevel=2,
        )


def get_data(name):
    """Return the local path to a lab data file, downloading it first if needed.

    The file's fingerprint is checked every time, so you always analyse exactly
    the data the lab was written for.
    """
    if name not in CHECKSUMS:
        raise KeyError(f"Unknown data file {name!r}. Available: {', '.join(CHECKSUMS)}")
    path = DATA_DIR / name
    if not path.exists():
        if not LAB_BASE_URL:
            raise FileNotFoundError(f"{path} is missing and the lab has no download URL configured.")
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"{LAB_BASE_URL}/data/{name}", path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKSUMS[name]:
        raise ValueError(f"{name} is not the file this lab expects. Ask a TA before continuing.")
    return path


def use_lab_style():
    """Apply the lab's plotting style to every figure, and show tables as plain numbers."""
    _check_versions()
    # Plain two-decimal numbers instead of scientific notation such as 2.574398e+04.
    pd.set_option("display.float_format", "{:.2f}".format)
    sns.set_palette([BLUE, ORANGE, AQUA])
    mpl.rcParams.update({
        "figure.figsize": (8, 4.5), "figure.dpi": 110, "savefig.dpi": 110,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.labelsize": 10, "axes.titlesize": 10.5,
        "axes.titlelocation": "left", "axes.titlecolor": INK_2, "axes.titlepad": 8,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA]),
        "lines.linewidth": 2, "lines.markersize": 8, "scatter.edgecolors": SURFACE,
        "patch.edgecolor": SURFACE, "legend.frameon": False,
    })


def finish(fig, title, subtitle=None, source=None):
    """Give a figure a headline that states the finding, a subtitle and a source line."""
    height = fig.get_size_inches()[1]
    top = 1 - (0.85 if subtitle else 0.55) / height
    bottom = 0.3 / height if source else 0
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.text(0.012, 1 - 0.12 / height, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.012, 1 - 0.45 / height, subtitle, ha="left", va="top", fontsize=10, color=INK_2)
    if source:
        fig.text(0.012, 0.08 / height, source, ha="left", va="bottom", fontsize=8, color=MUTED)
    return fig


def compare(**subplot_kw):
    """Side-by-side panels: the naive version (orange) left, the EDA version (blue) right."""
    fig, (naive, fixed) = plt.subplots(1, 2, figsize=(10, 4.5), **subplot_kw)
    naive.set_title("Skipping EDA", color=ORANGE, fontweight="semibold")
    fixed.set_title("After EDA", color=BLUE, fontweight="semibold")
    return fig, naive, fixed


def label(ax, x, y, text, color=INK_2, dx=6, dy=4, **kwargs):
    """Label one point directly, with a halo so the text stays readable over other points."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=color,
                path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)], **kwargs)


# ---------------------------------------------------------------------------
# Models used in the lab. The target is the access column, "electricity_access_pct" (or
# "electricity_access" or "access" in older versions of the notebook), from 0 to 100.
#   "everyone 100"  predicts 100% for every country, a yardstick to beat
#   "line"          a straight line fitted by least squares
#   "line, capped"  the same line, with predictions above 100 cut to 100
#   "s-curve"       a logistic curve fitted to the shares (a fractional logit),
#                   which bends towards 0 and 100 and never goes past them
# ---------------------------------------------------------------------------
MODELS = ("everyone 100", "line", "line, capped", "s-curve")


def _target(table):
    """The name of the electricity-access column in `table`."""
    for name in ("electricity_access_pct", "electricity_access"):
        if name in table.columns:
            return name
    return "access"


class Fitted:
    """A fitted model that can predict access for any table with the same feature columns."""

    def __init__(self, table, features, model):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.features, self.model = list(features), model
        if model == "everyone 100":
            return
        X = sm.add_constant(table[self.features].astype(float), has_constant="add")
        if model == "s-curve":
            family = sm.families.Binomial()
            self.result = sm.GLM(table[_target(table)] / 100, X, family=family).fit()
        else:
            self.result = sm.OLS(table[_target(table)], X).fit()

    def predict(self, table):
        if self.model == "everyone 100":
            return pd.Series(100.0, index=table.index)
        X = sm.add_constant(table[self.features].astype(float), has_constant="add")
        predicted = self.result.predict(X)
        if self.model == "s-curve":
            predicted = predicted * 100
        if self.model == "line, capped":
            predicted = predicted.clip(upper=100)
        return pd.Series(np.asarray(predicted), index=table.index)


def fit(table, features, model="s-curve"):
    """Fit one model to every row of `table` and return it (use .predict(new_table))."""
    return Fitted(table, features, model)


PERCENTAGE_COLUMNS = {"access", "cooking", "renewable_share",
                      "electricity_access", "clean_cooking_access", "renewable_energy_share",
                      "electricity_access_pct", "clean_cooking_access_pct",
                      "renewable_energy_share_pct"}


def plot_each_country(table, column):
    """Draw one small line plot of `column` from 2000 to 2020 for every country in `table`.

    Percentage columns share a 0 to 100 scale, so that a change from 100 to 99.7 looks as small
    as it is. Other columns use each country's own scale, because their levels differ by
    hundreds of times between countries.
    """
    shared = column in PERCENTAGE_COLUMNS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # tight_layout cannot fit 175 tiny panels
        grid = sns.relplot(data=table, x="Year", y=column, col="Entity", col_wrap=16,
                           kind="line", height=0.8, aspect=1.3, color=GREY, linewidth=1,
                           facet_kws={"sharey": shared, "sharex": True})
    grid.set_titles("{col_name}", size=5.5, pad=2)
    grid.set(xticks=[], yticks=[], xlabel="", ylabel="")
    if shared:
        grid.set(ylim=(0, 102))
    scale = "shared 0 to 100 scale" if shared else "each country on its own scale"
    grid.figure.suptitle(f"{column}, 2000 to 2020, one panel per country ({scale})", y=1.0)
    grid.figure.subplots_adjust(hspace=0.6, wspace=0.15)
    plt.show()


UNITS = {
    "gdp_per_capita": "GDP per capita (current US$)",
    "electricity_access": "Access to electricity (% of population)",
    "primary_energy_per_capita": "Primary energy per capita (kWh per person)",
    "clean_cooking_access": "Access to clean fuels for cooking (% of population)",
    "renewable_energy_share": "Renewable share of total final energy consumption (%)",
    "population_density": "Population density (people per km²)",
}


LABELS = {
    "gdp_per_capita": "GDP per capita", "electricity_access": "Electricity access",
    "primary_energy_per_capita": "Primary energy per capita", "clean_cooking_access": "Clean cooking access",
    "renewable_energy_share": "Renewable energy share", "population_density": "Population density",
    "log_gdp_per_capita": "log GDP per capita", "log_primary_energy_per_capita": "log primary energy per capita",
    "log_population_density": "log population density",
}
UNIT_ONLY = {
    "gdp_per_capita": "current US$ per person", "electricity_access": "% of population",
    "primary_energy_per_capita": "kWh per person", "clean_cooking_access": "% of population",
    "renewable_energy_share": "% of final energy consumption", "population_density": "people per km²",
}
# The lab names its percentage columns with a _pct suffix; earlier versions did not
for _names in (UNITS, LABELS, UNIT_ONLY):
    _names.update({f"{name}_pct": _names[name] for name in
                   ("electricity_access", "clean_cooking_access", "renewable_energy_share")})
COMMAS = mpl.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}")  # tick labels such as 10,000


def plot_fitted_models(table, models):
    """Draw each fitted model over the observations on a log GDP per capita axis, with its in-sample R²."""
    from matplotlib.ticker import FuncFormatter
    grid = pd.DataFrame({"gdp_per_capita": np.geomspace(table["gdp_per_capita"].min(),
                                                        table["gdp_per_capita"].max(), 300)})
    grid["log_gdp_per_capita"] = np.log10(grid["gdp_per_capita"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(table["gdp_per_capita"], table[_target(table)], s=12, color=GREY)
    for (name, model), colour in zip(models.items(), [ORANGE, AQUA, BLUE]):
        ax.plot(grid["gdp_per_capita"], model.predict(grid), color=colour,
                label=f"{name} (R² = {r_squared(table, model):.2f})")
    ax.axhline(100, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.set(xlabel="GDP per capita (current US$, log scale)", ylabel="Access to electricity (%)",
           title=f"Access against GDP per capita, 2019 (n = {len(table)})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)  # below the axes, clear of the points
    plt.show()


def outside_range(table, models):
    """Count each model's predictions below 0% or above 100% for the rows of `table`."""
    counts = {name: int(((m.predict(table) > 100) | (m.predict(table) < 0)).sum()) for name, m in models.items()}
    return pd.Series(counts, name="predictions outside 0-100%")


def plot_region_residuals(table, residual="residual", region="region", title="Residuals by region"):
    """Box plot of residuals by region with every country shown; returns each region's count and median."""
    by_region = table.groupby(region)[residual].agg(["count", "median"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = by_region.index
    sns.boxplot(data=table, y=region, x=residual, order=order, color="white", showfliers=False, ax=ax)
    sns.stripplot(data=table, y=region, x=residual, order=order, color=BLUE, size=3.5, ax=ax)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(range(len(order)), [f"{name} (n = {by_region.loc[name, 'count']})" for name in order])
    ax.set(xlabel="Residual: observed minus predicted access (percentage points)", ylabel="")
    ax.set_title(title, pad=18)
    above = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(-1, 1.01, "← model overestimates", transform=above, ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.text(1, 1.01, "model underestimates →", transform=above, ha="left", va="bottom", fontsize=8, color=INK_2)
    plt.show()
    return by_region


def add_one_at_a_time(table, base, others, regions, base_label="base model only", model="s-curve",
                      all_label="+ all of them (for context)", base_name=None, include_all=True):
    """Fit the base model, each single addition (and optionally all additions); report R² and regional median residuals.

    With `base_name`, each single-addition row is labelled "<base_name> + <measurement>", so the
    label says exactly which predictors that model contains.
    """
    def row_label(name):
        return f"{base_name} + {LABELS.get(name, name).lower()}" if base_name else f"+ {name}"
    candidates = {base_label: [], **{row_label(name): [name] for name in others}}
    if include_all:
        candidates[all_label] = others
    rows = {}
    for label, extra in candidates.items():
        fitted = fit(table, base + extra, model=model)
        residual = (table[_target(table)] - fitted.predict(table)).groupby(table["region"]).median()
        rows[label] = {"R²": r_squared(table, fitted),
                       **{f"{region} median residual": residual[region] for region in regions}}
    return pd.DataFrame(rows).T


def r_squared_table(table, models, model="s-curve"):
    """In-sample R² for each named list of predictors, shown to three decimal places."""
    return pd.Series({label: r_squared(table, fit(table, columns, model=model))
                      for label, columns in models.items()}, name="R²").map("{:.3f}".format)


def r_squared(table, model_fit):
    """Share of the variation in access across the rows of `table` that a fitted model captures.

    1 means every prediction is exact; 0 means the model does no better than predicting the
    average access for every country.
    """
    predicted = model_fit.predict(table)
    observed = table[_target(table)]
    residual = ((observed - predicted) ** 2).sum()
    total = ((observed - observed.mean()) ** 2).sum()
    return 1 - residual / total


def score(table, features, model="s-curve", seed=0):
    """Average error of a model on countries it was not fitted to (ten-fold cross-validation).

    Each country is predicted by a model fitted to the other nine tenths of the table.
    Returns the average error in percentage points for all countries, the same for the
    countries below 95% access, and how many predictions went above 100%. Every model uses
    the same ten groups for a given `seed`, so their scores can be compared; changing the seed
    shows how much the scores move by chance.
    """
    table = table.dropna(subset=["access", *features])
    predicted = pd.Series(np.nan, index=table.index)
    for train, test in KFold(n_splits=10, shuffle=True, random_state=seed).split(table):
        model_fit = Fitted(table.iloc[train], features, model)
        predicted.iloc[test] = model_fit.predict(table.iloc[test]).to_numpy()
    error = (predicted - table["access"]).abs()
    below = table["access"] < 95
    return pd.Series({
        "error, all countries": round(error.mean(), 1),
        "error, below 95%": round(error[below].mean(), 1),
        "predictions above 100%": int((predicted > 100.05).sum()),
    }, name=model if not features else f"{model}: {' + '.join(features)}", dtype=object)


def dollar_axis(ax, axis="x"):
    """Label a log10 GDP axis in dollars, so 3 reads as $1,000."""
    ticks = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    ticks = [t for t in ticks if lo <= np.log10(t) <= hi]
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    setter(np.log10(ticks), [f"${t:,}" for t in ticks])
    label = "GDP per person (current US$, log scale)"
    ax.set_xlabel(label) if axis == "x" else ax.set_ylabel(label)


def plot_fit(table, fits, x="log_gdp", ax=None):
    """Scatter of access against `x`, with the curve of each fitted model drawn on top.

    `fits` maps a label to a model fitted with `fit()` on the single feature `x`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(table[x], table["access"], s=18, color=GREY, alpha=0.7, edgecolors="none")
    grid = pd.DataFrame({x: np.linspace(table[x].min(), table[x].max(), 200)})
    for (name, model_fit), colour in zip(fits.items(), [ORANGE, BLUE, AQUA]):
        ax.plot(grid[x], model_fit.predict(grid), color=colour, linewidth=2.2, label=name)
    ax.axhline(100, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_ylabel("Access to electricity (%)")
    if x == "log_gdp":
        dollar_axis(ax)
    else:
        ax.set_xlabel(x)
    ax.legend(loc="lower right")
    return ax


def plot_residuals(table, expected, x="log_gdp", by=None, ax=None):
    """Plot actual minus expected access against `x`, optionally coloured by a yes/no column."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    residual = table["access"] - expected
    if by is None:
        ax.scatter(table[x], residual, s=18, color=GREY, alpha=0.8, edgecolors="none")
    else:
        for value, colour, name in [(1, ORANGE, by), (0, BLUE, f"not {by}")]:
            rows = table[by].astype(int) == value
            ax.scatter(table.loc[rows, x], residual[rows], s=18, color=colour, alpha=0.8,
                       edgecolors="none", label=name)
        ax.legend(loc="lower right")
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_ylabel("Actual minus expected access (points)")
    if x == "log_gdp":
        dollar_axis(ax)
    else:
        ax.set_xlabel(x)
    return ax


def out_of_fold_predictions(table, features, model="s-curve", seed=0):
    """Predict each country with a model fitted without that country's fold (the folds `score` uses).

    Returns a Series aligned with `table`; rows missing a feature or the outcome are left empty.
    Observed access minus this prediction is the out-of-fold prediction error.
    """
    usable = table.dropna(subset=["access", *features])
    predicted = pd.Series(np.nan, index=table.index)
    for train, test in KFold(n_splits=10, shuffle=True, random_state=seed).split(usable):
        model_fit = Fitted(usable.iloc[train], features, model)
        predicted.loc[usable.index[test]] = model_fit.predict(usable.iloc[test]).to_numpy()
    return predicted
