from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_PERMUTATIONS = 2000
N_BOOTSTRAP = 500
ALPHA = 0.05
RANDOM_STATE = 0

TARGET = "salary_mid_year"
COL_FEATURE = "col_index"
NUMERIC_CONTROLS = ["experience_years_min"]
CATEGORICAL_CONTROLS = ["category", "seniority", "work_mode", "employment_type", "company_size_band"]
BINARY_EXTRA = ["skills_listed"]
BINARY_PREFIX = "skill_category_"


def load_cleaned(path: str | Path) -> pd.DataFrame:
    """read the cleaned, col_index-merged postings"""
    return pd.read_csv(path)

def restrict_to_known_col_index(df: pd.DataFrame) -> pd.DataFrame:
    """keep postings with a matched MERIC cost-of-living index"""
    return df.dropna(subset=[COL_FEATURE]).copy()

def get_binary_controls(df: pd.DataFrame) -> list[str]:
    """find the skill_category_* dummy columns, plus skills_listed"""
    return [c for c in df.columns if c.startswith(BINARY_PREFIX)] + BINARY_EXTRA

def build_design_matrix(df: pd.DataFrame, numeric_medians: pd.Series) -> tuple[np.ndarray, list[str]]:
    """one-hot encode controls plus col_index into a single design matrix with an intercept column"""
    numbers = df[NUMERIC_CONTROLS].fillna(numeric_medians)
    labels = df[CATEGORICAL_CONTROLS].fillna("Unknown")
    dummies = pd.get_dummies(labels, drop_first=True, dtype=float)
    binaries = df[get_binary_controls(df)].astype(float)

    design = pd.concat([numbers, dummies, binaries, df[[COL_FEATURE]]], axis=1)
    assert design.columns.is_unique, "design matrix has duplicate column names"
    design_matrix = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    return design_matrix, list(design.columns)

def fit_ols(design_matrix: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """ordinary least squares via lstsq, returns coefficients (intercept first)"""
    return np.linalg.lstsq(design_matrix, outcome, rcond=None)[0]

def freedman_lane_test(
    full_matrix: np.ndarray,
    outcome: np.ndarray,
    col_position: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """freedman-lane permutation test for the coefficient at col_position, controlling for every other column (from slides)
    """
    reduced_matrix = np.delete(full_matrix, col_position, axis=1)
    reduced_pinv = np.linalg.pinv(reduced_matrix)
    fitted_reduced = reduced_matrix @ (reduced_pinv @ outcome)
    residuals = outcome - fitted_reduced

    full_pinv = np.linalg.pinv(full_matrix)
    observed = (full_pinv @ outcome)[col_position]

    permuted = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted_target = fitted_reduced + rng.permutation(residuals)
        permuted[i] = (full_pinv @ permuted_target)[col_position]

    p_value = (np.sum(np.abs(permuted) >= abs(observed)) + 1) / (n_permutations + 1)
    return observed, p_value

def bootstrap_confidence_interval(
    full_matrix: np.ndarray,
    outcome: np.ndarray,
    col_position: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """bootstrap a 95% CI for the coefficient at col_position by resampling rows with replacement"""
    n = len(outcome)
    coefficients = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        rows = rng.integers(0, n, n)
        coefficients[i] = fit_ols(full_matrix[rows], outcome[rows])[col_position]
    return np.percentile(coefficients, 2.5), np.percentile(coefficients, 97.5)

def write_results(results: pd.DataFrame, output_path: str | Path) -> None:
    """write the hypothesis test results table to csv"""
    results.to_csv(output_path, index=False)

def main() -> None:
    """load the cleaned data, run the freedman-lane test and a bootstrap CI for col_index, and write results"""
    input_path = DATA_DIR / "cleaned_job_postings.csv"
    output_path = DATA_DIR / "cost_of_living_hypothesis_test.csv"

    df = load_cleaned(input_path)
    df_test = restrict_to_known_col_index(df)
    numeric_medians = df_test[NUMERIC_CONTROLS].median()

    full_matrix, columns = build_design_matrix(df_test, numeric_medians)
    outcome = df_test[TARGET].to_numpy()
    col_position = columns.index(COL_FEATURE) + 1  # +1 for the intercept column

    rng = np.random.default_rng(RANDOM_STATE)
    observed, p_value = freedman_lane_test(full_matrix, outcome, col_position, N_PERMUTATIONS, rng)
    ci_low, ci_high = bootstrap_confidence_interval(full_matrix, outcome, col_position, N_BOOTSTRAP, rng)

    results = pd.DataFrame([{
        "feature": COL_FEATURE,
        "n_postings": len(df_test),
        "coefficient": observed,
        "p_value": p_value,
        "significant": p_value < ALPHA,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }])

    write_results(results, output_path)
    print(f"Postings used (known col_index): {len(df_test)} of {len(df)}")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
