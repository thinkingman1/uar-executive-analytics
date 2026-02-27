import pandas as pd

def executive_summary(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum()) if rows > 0 else 0

    summary = {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "duplicate_rows": dup_rows,
    }

    # Date coverage if any date-like columns exist
    date_cols = []
    for c in df.columns:
        col = df[c]
        if not (
            pd.api.types.is_datetime64_any_dtype(col)
            or pd.api.types.is_object_dtype(col)
            or pd.api.types.is_string_dtype(col)
        ):
            continue

        # Sample first to avoid full-column conversion for clearly non-date text columns.
        sample = col.dropna().head(200)
        if sample.empty:
            continue
        sample_dt = pd.to_datetime(sample, errors="coerce")
        if sample_dt.notna().mean() < 0.6:
            continue

        s = pd.to_datetime(col, errors="coerce")
        if s.notna().mean() >= 0.6:  # mostly date-like
            date_cols.append((c, s))

    if date_cols:
        # pick the most date-like column
        best = max(date_cols, key=lambda x: x[1].notna().mean())
        c, s = best
        summary["date_column"] = c
        summary["date_min"] = str(s.min().date()) if pd.notna(s.min()) else ""
        summary["date_max"] = str(s.max().date()) if pd.notna(s.max()) else ""
    else:
        summary["date_column"] = ""
        summary["date_min"] = ""
        summary["date_max"] = ""

    return summary

def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    return miss.rename("missing_percent").to_frame()

def infer_key_columns(df: pd.DataFrame) -> dict:
    # Normalise column names
    cols = {c: str(c).strip().lower() for c in df.columns}

    def pick(candidates):
        for cand in candidates:
            for orig, low in cols.items():
                if cand in low:
                    return orig
        return None

    return {
        "employee": pick(["employee", "emp", "user", "userid", "user id"]),
        "manager": pick(["manager", "reports to", "reporting", "supervisor"]),
        "department": pick(["department", "dept", "function"]),
        "location": pick(["location", "city", "site"]),
        "status": pick(["status", "active", "employment status"]),
        "join_date": pick(["join", "joining", "hire", "start date"]),
        "exit_date": pick(["exit", "release", "end date", "termination"]),
    }
