from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_PERMUTATIONS = 2000
ALPHA = 0.005  # lower threshold to account for testing 10 categories
RANDOM_STATE = 0
STRATA_COLS = ["state", "seniority"]  # "region and seniority level" in the null hypothesis
CATEGORY_PREFIX = "skill_category_"


def load_preprocessed(path: str | Path) -> pd.DataFrame:
    """read the preprocessed jobs data (already has the skill_category_* dummies)"""
    return pd.read_csv(path)


def get_category_columns(df: pd.DataFrame) -> list[str]:
    """find every skill_category_* dummy column in the preprocessed data"""
    return [c for c in df.columns if c.startswith(CATEGORY_PREFIX)]


def restrict_to_strata(df: pd.DataFrame) -> pd.DataFrame:
    """keep postings with a known value for every stratifying column"""
    return df.dropna(subset=STRATA_COLS).copy()


def build_strata(df: pd.DataFrame) -> np.ndarray:
    """integer-code each posting's stratum, defined by STRATA_COLS"""
    codes, _ = pd.factorize(pd.Series(list(zip(*(df[col] for col in STRATA_COLS)))))
    return codes


def mean_difference(salary: np.ndarray, has_skill: np.ndarray) -> float:
    """difference in mean salary between postings with vs without the skill"""
    return salary[has_skill == 1].mean() - salary[has_skill == 0].mean()


def permute_within_strata(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """return `values` randomly shuffled within each stratum (same strata membership, random order)"""
    random_keys = rng.random(len(values))
    canonical_order = np.argsort(strata, kind="stable")
    shuffled_order = np.lexsort((random_keys, strata))
    shuffled = np.empty_like(values)
    shuffled[canonical_order] = values[shuffled_order]
    return shuffled


def stratified_permutation_test(
    salary: np.ndarray,
    has_skill: np.ndarray,
    strata: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """two-sided permutation test for a mean salary difference, permuting has_skill within strata"""
    observed = mean_difference(salary, has_skill)
    permuted = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_skill = permute_within_strata(has_skill, strata, rng)
        permuted[i] = mean_difference(salary, shuffled_skill)
    p_value = (np.sum(np.abs(permuted) >= abs(observed)) + 1) / (n_permutations + 1)
    return observed, p_value


def run_all_tests(
    df: pd.DataFrame,
    salary: np.ndarray,
    strata: np.ndarray,
    category_cols: list[str],
    n_permutations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """run the stratified permutation test for every skill category column"""
    rows = []
    for col in category_cols:
        has_skill = df[col].to_numpy()
        observed, p_value = stratified_permutation_test(salary, has_skill, strata, n_permutations, rng)
        rows.append({
            "category": col,
            "n_with_skill": int(has_skill.sum()),
            "n_without_skill": int((has_skill == 0).sum()),
            "mean_salary_diff": observed,
            "p_value": p_value,
            "significant": p_value < ALPHA,
        })
    return pd.DataFrame(rows)


def write_results(results: pd.DataFrame, output_path: str | Path) -> None:
    """write the hypothesis test results table to csv"""
    results.to_csv(output_path, index=False)


def main() -> None:
    input_path = DATA_DIR / "preprocessed_jobs.csv"
    output_path = DATA_DIR / "skill_salary_hypothesis_tests.csv"

    df = load_preprocessed(input_path)
    df_test = restrict_to_strata(df)
    strata = build_strata(df_test)
    salary = df_test["salary_mid_year"].to_numpy()
    category_cols = get_category_columns(df_test)

    rng = np.random.default_rng(RANDOM_STATE)
    results = run_all_tests(df_test, salary, strata, category_cols, N_PERMUTATIONS, rng)

    write_results(results, output_path)
    print(f"Rows used (known state & seniority): {len(df_test)} of {len(df)}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
