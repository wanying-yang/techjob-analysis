import ast
import re
from collections import Counter
from pathlib import Path

import geonamescache
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
import us
from sklearn.decomposition import NMF

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_CATEGORIES = 9
TOP_N_SKILLS = 150
RANDOM_STATE = 0

SALARY_COLS = ["salary_min", "salary_max", "salary_currency", "salary_period"]
PREDICTOR_COLS = [
    "title", "work_mode", "skills", "seniority", "category", "employment_type",
    "experience_years_min", "company_industries", "company_types", "company_sizes",
]
# for time conversions
SALARY_MULTIPLIER = {"year": 1, "month": 12, "day": 260, "hour": 2080}

LOC_COL = "location"
OTHER_LABEL = "untraditional"  # label for locations that name a site, not a place
EXTRA_CITIES = {
    "nyc": "NY", "bay area": "CA", "silicon valley": "CA",
    "dmv": "DC", "research triangle": "NC",
}


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """read the raw freehire_tech.csv"""
    return pd.read_csv(csv_path)


def filter_us_jobs(data: pd.DataFrame) -> pd.DataFrame:
    """keep postings whose countries list is exactly ['us']"""
    return data[data["countries"].apply(lambda x: ast.literal_eval(x) == ["us"])].copy()


def filter_2026_jobs(us_jobs: pd.DataFrame) -> pd.DataFrame:
    """parse posted_at and keep postings from 2026"""
    us_jobs = us_jobs.copy()
    us_jobs["posted_at"] = pd.to_datetime(us_jobs["posted_at"], format="ISO8601", utc=True)
    return us_jobs[us_jobs["posted_at"].dt.year == 2026]


def filter_salary_usd(us_jobs_2026: pd.DataFrame) -> pd.DataFrame:
    """drop postings missing salary fields and keep usd-only postings"""
    withsalary = us_jobs_2026.dropna(subset=SALARY_COLS).copy()
    withsalary["salary_currency"] = withsalary["salary_currency"].str.upper().str.strip()
    return withsalary[withsalary["salary_currency"] == "USD"].copy()


def compute_salary_mid_year(df: pd.DataFrame) -> pd.DataFrame:
    """annualize `salary_min`/`salary_max` and add `salary_mid_year`"""
    df = df.copy()
    factor = df["salary_period"].map(SALARY_MULTIPLIER)
    df["salary_min_year"] = df["salary_min"] * factor
    df["salary_max_year"] = df["salary_max"] * factor
    df["salary_mid_year"] = (df["salary_min_year"] + df["salary_max_year"]) / 2
    return df


def build_location_lookups() -> dict:
    """build the state-name/city/regex lookups used by find_state"""
    state_names = {s.name.lower(): s.abbr for s in us.states.STATES + [us.states.DC]}
    state_abbrs = set(state_names.values())

    gc = geonamescache.GeonamesCache()
    us_cities = sorted(
        (c for c in gc.get_cities().values() if c["countrycode"] == "US"),
        key=lambda c: c["population"],
    )
    city_to_state = {c["name"].lower(): c["admin1code"] for c in us_cities}
    city_to_state.update(EXTRA_CITIES)

    state_name_re = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, state_names), key=len, reverse=True)) + r")\b",
        re.I,
    )
    abbr_re = re.compile(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])")
    remote_re = re.compile(r"\b(remote|anywhere|work from home|wfh)\b", re.I)
    nation_re = re.compile(r"\b(nationwide|multiple locations|various locations)\b", re.I)
    country_re = re.compile(r"\b(united states( of america)?|usa|u\.s\.a?\.?|us)\b", re.I)
    area_re = re.compile(r"^greater\s+|\s+(bay|metro(politan)?)?\s*area$", re.I)

    return {
        "state_names": state_names,
        "state_abbrs": state_abbrs,
        "city_to_state": city_to_state,
        "state_name_re": state_name_re,
        "abbr_re": abbr_re,
        "remote_re": remote_re,
        "nation_re": nation_re,
        "country_re": country_re,
        "area_re": area_re,
    }


def split_parts(loc: str) -> list[str]:
    """'Remote - US' -> ['Remote', 'US']"""
    loc = re.sub(r"\s*[-–—|/;()]\s*", ",", loc)
    return [p.strip() for p in loc.split(",") if p.strip()]


def find_state(loc: str, lookups: dict) -> str | None:
    """resolve a free-text location string to a state abbreviation"""
    state_names, state_abbrs = lookups["state_names"], lookups["state_abbrs"]
    city_to_state, area_re = lookups["city_to_state"], lookups["area_re"]
    abbr_re, state_name_re = lookups["abbr_re"], lookups["state_name_re"]

    parts = split_parts(loc)

    # 1. a whole part is a state abbreviation or name ("Seattle, WA")
    for p in reversed(parts):
        if p in state_abbrs:
            return p
        if p.lower() in state_names:
            return state_names[p.lower()]

    # 2. a whole part is a known city or metro area ("San Francisco")
    for p in parts:
        for key in (p.lower(), area_re.sub("", p).lower()):
            if key in city_to_state:
                return city_to_state[key]

    # 3. state abbreviation or name inside the text ("Atlanta GA", "USA.VA.Reston")
    abbrs = [a for a in abbr_re.findall(loc) if a in state_abbrs]
    if abbrs:
        return abbrs[-1]
    m = state_name_re.search(loc)
    if m:
        return state_names[m.group(1).lower()]

    # 4. city name inside the text ("Denver HQ"), longest phrase first
    words = re.findall(r"[a-z]+", loc.lower())
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if phrase in city_to_state:
                return city_to_state[phrase]
    return None


def location_type(loc: object, state: object, lookups: dict) -> str:
    """classify a location as state / remote / us_only / untraditional / missing"""
    remote_re, nation_re, country_re = lookups["remote_re"], lookups["nation_re"], lookups["country_re"]
    if pd.notna(state):
        return "state"
    if not isinstance(loc, str) or not loc.strip():
        return "missing"
    if remote_re.search(loc):
        return "remote"
    if nation_re.search(loc) or not re.sub(r"[\W_]+", "", country_re.sub("", loc)):
        return "us_only"
    return OTHER_LABEL


def add_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """add state and location_type columns, resolving each unique location once"""
    df = df.copy()
    lookups = build_location_lookups()
    unique_locs = df[LOC_COL].dropna().unique()
    state_lookup = {loc: find_state(loc, lookups) for loc in unique_locs}

    df["state"] = df[LOC_COL].map(state_lookup)
    df["location_type"] = [
        location_type(loc, state, lookups) for loc, state in zip(df[LOC_COL], df["state"])
    ]
    return df


def parse_skills_list(df: pd.DataFrame) -> pd.DataFrame:
    """add a skills_list column parsed from the skills string column"""
    df = df.copy()
    df["skills_list"] = df["skills"].apply(ast.literal_eval)
    return df


def get_top_skills(df: pd.DataFrame, top_n: int) -> list[str]:
    """return the top_n most frequent skills across all postings"""
    skill_counts = df["skills_list"].explode().value_counts()
    return list(skill_counts.head(top_n).index)


def build_skill_matrix(df: pd.DataFrame, top_skills: list[str]) -> sp.csr_matrix:
    """build a sparse postings by skills binary one-hot matrix"""
    skill_idx = {skill: i for i, skill in enumerate(top_skills)}
    rows, cols = [], []
    for i, skills in enumerate(df["skills_list"]):
        for skill in skills:
            j = skill_idx.get(skill)
            if j is not None:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(len(df), len(top_skills)))


def cluster_skills(
    matrix: sp.csr_matrix, top_skills: list[str], n_clusters: int
) -> dict[int, list[str]]:
    """nmf-decompose postings by skills into topics, hard-assigning each skill to its top topic"""
    nmf = NMF(n_components=n_clusters, init="nndsvda", random_state=RANDOM_STATE, max_iter=400)
    nmf.fit(matrix)
    top_topic = np.argmax(nmf.components_, axis=0)
    clusters: dict[int, list[str]] = {c: [] for c in range(n_clusters)}
    for skill, topic in zip(top_skills, top_topic):
        clusters[topic].append(skill)
    return clusters


def write_clusters(clusters: dict[int, list[str]], output_path: str | Path) -> None:
    """write categories out to a file for manual inspection"""
    with open(output_path, "w") as f:
        for category, skills in clusters.items():
            f.write(f"CATEGORY {category}\n")
            for skill in skills:
                f.write(f"{skill}\n")
            f.write("\n")


def add_cluster_dummies(df: pd.DataFrame, clusters: dict[int, list[str]]) -> pd.DataFrame:
    """add one skill_category_<i> dummy per cluster, plus skills_listed"""
    df = df.copy()
    for category, skills in clusters.items():
        skill_set = set(skills)
        df[f"skill_category_{category}"] = df["skills_list"].apply(
            lambda lst: int(any(s in skill_set for s in lst))
        )
    df["skills_listed"] = (df["skills_list"].str.len() > 0).astype(int)
    return df


def quick_checks(df: pd.DataFrame, skill_cols: list[str]) -> None:
    """print category share and median salary with/without each category"""
    print("Share of jobs (that list skills) in each category:")
    print(df.loc[df["skills_listed"] == 1, skill_cols].mean().round(3), "\n")

    print("Median salary without (0) vs with (1) each category:")
    for c in skill_cols:
        print(c, df.groupby(c)["salary_mid_year"].median().to_dict())


def main() -> None:
    """run the full preprocessing pipeline and write out the skill clusters"""
    csv_path = DATA_DIR / "freehire_tech.csv"
    output_path = DATA_DIR / "skill_categories.txt"

    data = load_data(csv_path)
    us_jobs = filter_us_jobs(data)
    us_jobs_2026 = filter_2026_jobs(us_jobs)
    us_jobs_2026_withsalary = filter_salary_usd(us_jobs_2026)
    df = compute_salary_mid_year(us_jobs_2026_withsalary)
    df = add_state_columns(df)
    df = parse_skills_list(df)

    top_skills = get_top_skills(df, TOP_N_SKILLS)
    matrix = build_skill_matrix(df, top_skills)
    clusters = cluster_skills(matrix, top_skills, N_CATEGORIES)
    write_clusters(clusters, output_path)

    df = add_cluster_dummies(df, clusters)
    skill_cols = [f"skill_category_{i}" for i in range(N_CATEGORIES)]
    quick_checks(df, skill_cols)


if __name__ == "__main__":
    main()
