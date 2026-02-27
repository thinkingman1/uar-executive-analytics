import pandas as pd

def mermaid_org_chart(df: pd.DataFrame, employee_col: str, manager_col: str, limit: int = 200) -> str:
    d = df[[employee_col, manager_col]].copy()
    d[employee_col] = d[employee_col].astype(str).str.strip()
    d[manager_col] = d[manager_col].astype(str).str.strip()

    d = d.dropna()
    d = d[(d[employee_col] != "") & (d[manager_col] != "")]
    d = d.head(limit)

    # Mermaid-safe ids
    def mid(x: str) -> str:
        return "N_" + "".join(ch if ch.isalnum() else "_" for ch in x)[:50]

    lines = ["flowchart TB"]
    seen = set()

    for emp, mgr in d[[employee_col, manager_col]].itertuples(index=False, name=None):
        emp = str(emp)
        mgr = str(mgr)
        e_id, m_id = mid(emp), mid(mgr)

        if e_id not in seen:
            lines.append(f'{e_id}["{emp}"]')
            seen.add(e_id)
        if m_id not in seen:
            lines.append(f'{m_id}["{mgr}"]')
            seen.add(m_id)

        lines.append(f"{m_id} --> {e_id}")

    return "\n".join(lines)
