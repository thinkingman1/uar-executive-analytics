import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
from io import BytesIO

st.set_page_config(page_title="UAR Executive Analytics", layout="wide")

st.title("UAR Executive Analytics")
st.caption("Upload a CSV or Excel file to view analytics for executive review.")

# =========================
# Theme selector
# =========================
theme_name = st.selectbox(
    "Theme",
    ["Light", "Dark", "Slate", "Navy"],
    index=0,
    help="Changes background and chart theme for executive viewing.",
)

THEMES = {
    "Dark": {
        "bg": "#0E1117",
        "fg": "#FAFAFA",
        "card": "#111827",
        "grid": "rgba(255,255,255,0.08)",
        "plotly_template": "plotly_dark",
    },
    "Light": {
        "bg": "#FFFFFF",
        "fg": "#111111",
        "card": "#F6F7F9",
        "grid": "rgba(0,0,0,0.10)",
        "plotly_template": "plotly_white",
    },
    "Slate": {
        "bg": "#0B1220",
        "fg": "#E5E7EB",
        "card": "#0F172A",
        "grid": "rgba(255,255,255,0.08)",
        "plotly_template": "plotly_dark",
    },
    "Navy": {
        "bg": "#071226",
        "fg": "#EAF2FF",
        "card": "#0B1B3A",
        "grid": "rgba(255,255,255,0.10)",
        "plotly_template": "plotly_dark",
    },
}

# Default fallback is Light
THEME = THEMES.get(theme_name, THEMES["Light"])

st.markdown(
    f"""
    <style>
      .stApp {{
        background-color: {THEME['bg']};
        color: {THEME['fg']};
      }}
      .stMarkdown, .stText, .stCaption, .stDataFrame {{
        color: {THEME['fg']} !important;
      }}
      div[data-testid="stRadio"] label p,
      div[data-testid="stRadio"] label,
      div[data-testid="stMetricLabel"] p,
      div[data-testid="stMetricValue"] {{
        color: {THEME['fg']} !important;
      }}
      /* Make tables look cleaner on dark themes */
      div[data-testid="stDataFrame"] {{
        background-color: transparent;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper to apply theme to Plotly figures
def apply_plotly_theme(fig):
    """Apply the selected theme to a Plotly figure (background, fonts, axes, legend)."""
    if fig is None:
        return fig

    # Template controls many defaults, but we also pin bg/font/grid for consistency.
    fig.update_layout(
        template=THEME["plotly_template"],
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["fg"]),
        legend=dict(font=dict(color=THEME["fg"])),
    )

    # Axes styling (safe even for charts without axes)
    fig.update_xaxes(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        linecolor=THEME["grid"],
        tickfont=dict(color=THEME["fg"]),
        title_font=dict(color=THEME["fg"]),
    )
    fig.update_yaxes(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        linecolor=THEME["grid"],
        tickfont=dict(color=THEME["fg"]),
        title_font=dict(color=THEME["fg"]),
    )
    return fig


def get_col_by_excel_letter(df: pd.DataFrame, letter: str):
    """Maps Excel-like column letters to dataframe column by position."""
    if df is None:
        return None
    letter = letter.strip().upper()
    idx = ord(letter) - ord("A")
    if idx < 0 or idx >= df.shape[1]:
        return None
    return df.columns[idx]


@st.cache_data(show_spinner=False)
def load_file_from_bytes(file_bytes: bytes, name: str) -> pd.DataFrame:
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    return pd.read_excel(BytesIO(file_bytes))


def load_file(file):
    return load_file_from_bytes(file.getvalue(), file.name.lower())


def to_bool_series_any(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .fillna(False)
    )


# =========================
# Column mapping (robust to new/reordered columns)
# =========================

FIELD_SPECS = {
    # field_key: {label, required, aliases}
    "identity": {
        "label": "Identity (used for total identities)",
        "required": True,
        "aliases": [
            "identity",
            "user",
            "user id",
            "userid",
            "account",
            "account id",
            "accountid",
            "identity name",
            "name",
            "employee",
            "employee id",
            "certification_id",
            "certification id",
            "certification",
            "record id",
            "id",
        ],
    },
    "certifier": {
        "label": "Certifier (initial certifier)",
        "required": True,
        "aliases": ["certifier", "certifier name", "reviewer", "approver", "owner", "manager", "certifier identity"],
    },
    "reassigned_flag": {
        "label": "Reassigned flag (TRUE/FALSE)",
        "required": True,
        "aliases": ["reassigned", "is reassigned", "reassigned?", "reassignment", "reassigned flag", "reassigned true"],
    },
    "reassigned_to": {
        "label": "Reassigned to (identity)",
        "required": False,
        "aliases": ["reassigned to", "reassigned identity", "reassigned user", "delegate", "delegated to", "assigned to", "new certifier"],
    },
    "actor": {
        "label": "Final actor (who approved/revoked)",
        "required": False,
        "aliases": ["actor", "final actor", "actioned by", "acted by", "performed by", "last updated by", "action by"],
    },
    "status": {
        "label": "Status (Approve/Revoke)",
        "required": True,
        "aliases": ["status", "decision", "action", "outcome", "certification decision"],
    },
    "updated_at": {
        "label": "Updated at (action timestamp)",
        "required": False,
        "aliases": ["updated at", "updated_at", "action date", "action time", "decision date", "last updated", "modified at", "completed at"],
    },
    "activated_at": {
        "label": "Activated at (campaign start)",
        "required": False,
        "aliases": ["activated at", "activated_at", "start date", "campaign start", "campaign activated", "begin date"],
    },
    "end_date": {
        "label": "End date (campaign end)",
        "required": False,
        "aliases": ["end date", "end_date", "campaign end", "expiry", "expires at", "close date"],
    },
}

def _normalize_colname(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

def autodetect_columns(df: pd.DataFrame) -> dict:
    """Return a mapping field_key -> column_name using header alias matching."""
    cols_norm = {_normalize_colname(c): c for c in df.columns}
    mapping = {}
    for field, spec in FIELD_SPECS.items():
        found = None
        for a in spec["aliases"]:
            a_norm = _normalize_colname(a)
            if a_norm in cols_norm:
                found = cols_norm[a_norm]
                break
        mapping[field] = found
    # Fallback: if identity was not auto-detected, use the first column as a best-effort guess.
    # The user can still override this in the sidebar mapping UI.
    if mapping.get("identity") is None and df is not None and df.shape[1] > 0:
        mapping["identity"] = df.columns[0]
    return mapping

def mapping_ui(df: pd.DataFrame, auto_map: dict) -> dict:
    """Sidebar UI to confirm/override mapping. Returns final mapping."""
    with st.sidebar:
        st.markdown("## Column mapping")
        st.caption("If the file format changes, update mappings once here. The rest of the dashboard adapts.")

        final = {}
        for field, spec in FIELD_SPECS.items():
            options = ["(Not set)"] + list(df.columns)
            default = auto_map.get(field)
            default_idx = options.index(default) if default in options else 0

            final[field] = st.selectbox(
                spec["label"],
                options=options,
                index=default_idx,
                key=f"map_{field}",
            )
            if final[field] == "(Not set)":
                final[field] = None

        return final

def require_mapping_ok(mapping: dict) -> list:
    missing = []
    for field, spec in FIELD_SPECS.items():
        if spec.get("required") and not mapping.get(field):
            missing.append(field)
    return missing


uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Please upload a file to continue.")
    st.stop()

try:
    df = load_file(uploaded)
    raw_df = df.copy()
except Exception as e:
    st.error("Unable to read the uploaded file.")
    st.exception(e)
    st.stop()

st.subheader("Preview")

# =========================
# Sidebar filters (Excel-like)
# =========================
with st.sidebar:
    st.markdown("### Preview filters")

    # Choose a column to filter
    filter_col = st.selectbox(
        "Filter column",
        options=["None"] + list(raw_df.columns),
        index=0,
    )

    # Initialize session state for filters
    if "preview_filter" not in st.session_state:
        st.session_state.preview_filter = {
            "col": "None",
            "values": None,
        }

    if filter_col == "None":
        st.info("Select a column to filter the preview.")
        search_text = ""
        selected_values = None
        st.markdown("**Selected values**")
        st.caption("Choose a filter column first")
    else:
        # Build unique value list for selected column
        col_series = raw_df[filter_col].fillna("").astype(str).str.strip()
        unique_vals = sorted(col_series.unique().tolist())

        search_text = st.text_input("Search values", value="")
        if search_text:
            unique_vals = [v for v in unique_vals if search_text.lower() in v.lower()]

        # Default selection: keep previous if same column, else select all
        prev = st.session_state.preview_filter
        if prev.get("col") == filter_col and isinstance(prev.get("values"), list):
            default_vals = [v for v in prev["values"] if v in unique_vals]
            if len(default_vals) == 0:
                default_vals = unique_vals
        else:
            default_vals = unique_vals

        selected_values = st.multiselect(
            "Select values",
            options=unique_vals,
            default=default_vals,
            key="preview_filter_values",
            help="Start typing above to narrow the list. Selected values will be shown below.",
        )

        # Show selected values at the bottom (dynamic)
        st.markdown("**Selected values**")
        if selected_values:
            st.caption(f"{len(selected_values):,} selected")
            # Show as a compact, searchable list
            st.write(selected_values)
        else:
            st.caption("None selected")

    c1, c2 = st.columns(2)
    apply_clicked = c1.button("Apply filter")
    clear_clicked = c2.button("Clear filter")

    if clear_clicked:
        st.session_state.preview_filter = {"col": "None", "values": None}
        st.rerun()

    if apply_clicked:
        if filter_col == "None":
            st.session_state.preview_filter = {"col": "None", "values": None}
        else:
            st.session_state.preview_filter = {"col": filter_col, "values": selected_values}
        st.rerun()

# Apply sidebar filter to preview dataframe
pf = st.session_state.get("preview_filter", {"col": "None", "values": None})
preview_df = raw_df.copy()
if pf.get("col") != "None" and isinstance(pf.get("values"), list):
    col_name = pf["col"]
    sel = set([str(v).strip() for v in pf["values"]])
    s = preview_df[col_name].fillna("").astype(str).str.strip()
    preview_df = preview_df[s.isin(sel)]

st.caption(f"Preview rows: {len(preview_df):,} of {len(raw_df):,}")
st.dataframe(preview_df, use_container_width=True, height=320)

# Build column mapping (auto-detect + UI override)
auto_map = autodetect_columns(raw_df)
final_map = mapping_ui(raw_df, auto_map)

missing_required_fields = require_mapping_ok(final_map)
if missing_required_fields:
    missing_labels = [FIELD_SPECS[f]["label"] for f in missing_required_fields]
    st.error("Missing required column mappings: " + ", ".join(missing_labels))
    st.stop()

# =========================
# Analytics scope selector (Certification ID value)
# =========================
# Use the mapped Identity column as the certification identifier for scoping.
# If multiple certification IDs exist in one file, the user can pick which one to analyze.
df = raw_df.copy()

with st.sidebar:
    st.markdown("## Analytics scope")
    id_col = final_map.get("identity")

    selected_cert_id = None
    if id_col is not None and id_col in df.columns:
        id_series = df[id_col].dropna()
        # Keep native types where possible, but show friendly labels
        unique_ids = sorted(id_series.unique().tolist(), key=lambda x: str(x))

        if len(unique_ids) > 1:
            options = unique_ids
            selected_cert_id = st.selectbox(
                "Choose Certification ID value",
                options=options,
                index=0,
                help="If the file contains multiple Certification IDs, select one to scope all analytics.",
            )
            df = df[df[id_col] == selected_cert_id].copy()
            st.caption(f"Analytics scoped to {id_col} = {selected_cert_id} (rows: {len(df):,}).")
        else:
            st.caption("Single Certification ID detected. Using full file for analytics.")
    else:
        st.caption("Identity mapping not available for scoping.")

# If filtering results in no rows, stop early with a clear message
if df.empty:
    st.warning("No rows matched the selected Certification ID value. Please choose a different value.")
    st.stop()

# =========================
# Access Review KPIs
# =========================
st.markdown("### Access Review KPIs")

A = df[final_map["identity"]] if final_map.get("identity") else None
D = df[final_map["certifier"]] if final_map.get("certifier") else None
E = df[final_map["reassigned_flag"]] if final_map.get("reassigned_flag") else None
F = df[final_map["reassigned_to"]] if final_map.get("reassigned_to") else None
G = df[final_map["actor"]] if final_map.get("actor") else None
L = df[final_map["status"]] if final_map.get("status") else None
P = df[final_map["updated_at"]] if final_map.get("updated_at") else None
Q = df[final_map["activated_at"]] if final_map.get("activated_at") else None
R = df[final_map["end_date"]] if final_map.get("end_date") else None

# Total identities should be the total number of rows in the scoped dataset
total_identities = int(len(df))

# COUNTUNIQUE(FILTER(D, D<>""))
initial_certifier_count = 0
if D is not None:
    d_nonempty = D.dropna().astype(str).str.strip()
    d_nonempty = d_nonempty[d_nonempty != ""]
    initial_certifier_count = int(d_nonempty.nunique())

# COUNTIF(E, TRUE)
reassigned_count = int(to_bool_series_any(E).sum()) if E is not None else 0

# COUNTA(UNIQUE(FILTER(F, F<>"")))-1
reassigned_unique_identities = 0
if F is not None:
    f_nonempty = F.dropna().astype(str).str.strip()
    f_nonempty = f_nonempty[f_nonempty != ""]
    reassigned_unique_identities = max(int(f_nonempty.nunique()) - 1, 0)

# Unique reassigners / unique reassigned-to (based on reassignment cases)
unique_reassigners = 0
unique_reassigned_to = 0
if D is not None and E is not None and F is not None and len(df) > 0:
    e_true_kpi = to_bool_series_any(E)
    d_str_kpi = D.fillna("").astype(str).str.strip()
    f_str_kpi = F.fillna("").astype(str).str.strip()

    reassigned_mask_kpi = e_true_kpi & (f_str_kpi != "") & (d_str_kpi != f_str_kpi)

    unique_reassigners = int(d_str_kpi[reassigned_mask_kpi].nunique())
    unique_reassigned_to = int(f_str_kpi[reassigned_mask_kpi].nunique())

# SUMPRODUCT((E=TRUE)*(F=G)) etc
reassigned_and_final_same = 0
reassigned_final_diff_count = 0
reassigned_final_diff_unique_strings = 0
if E is not None and F is not None and G is not None and len(df) > 0:
    e_true = to_bool_series_any(E)
    f_str = F.fillna("").astype(str).str.strip()
    g_str = G.fillna("").astype(str).str.strip()

    reassigned_and_final_same = int((e_true & (f_str == g_str)).sum())
    reassigned_final_diff_count = int((e_true & (f_str != g_str)).sum())
    mask = e_true & (f_str != "") & (f_str != g_str)
    reassigned_final_diff_unique_strings = int(g_str[mask].nunique())

# Reassigned revoked by Citadel due to end of Campaign
# Excel logic:
# =SUMPRODUCT((E=TRUE)*(G="Super Admin")*(INT(P)=INT(R)))
reassigned_revoked_by_citadel_end_campaign = 0

# Total revoked by Citadel due to end of Campaign
# Excel logic:
# =SUMPRODUCT((G="Super Admin")*(IFERROR(INT(P),0)=IFERROR(INT(R),-1)))
total_revoked_by_citadel_end_campaign = 0

if G is not None and P is not None and R is not None and len(df) > 0:
    g_str = G.fillna("").astype(str).str.strip()
    p_dt = pd.to_datetime(P, errors="coerce")
    r_dt = pd.to_datetime(R, errors="coerce")
    same_day = (p_dt.dt.date == r_dt.dt.date).fillna(False)

    # Total revoked by Citadel due to end of Campaign (no reassigned flag condition)
    total_revoked_by_citadel_end_campaign = int(((g_str == "Super Admin") & same_day).sum())

    # Reassigned revoked by Citadel due to end of Campaign (requires reassigned_flag == TRUE)
    if E is not None:
        e_true = to_bool_series_any(E)
        reassigned_revoked_by_citadel_end_campaign = int((e_true & (g_str == "Super Admin") & same_day).sum())

# COUNTIF(L,"Approve") and COUNTIF(L,"Revoke")
total_approved = 0
total_revoked = 0
if L is not None:
    l_str = L.fillna("").astype(str).str.strip()
    total_approved = int((l_str == "Approve").sum())
    total_revoked = int((l_str == "Revoke").sum())

kpis = [
    ("Total identities", total_identities),
    ("Initial certifier count", initial_certifier_count),
    ("Reassigned count", reassigned_count),
    ("Reassigned unique identities", reassigned_unique_identities),
    ("Reassigned and final actor are same", reassigned_and_final_same),
    ("Reassigned but Final Actor is different count", reassigned_final_diff_count),
    ("Reassigned but Final Actor is different (unique strings)", reassigned_final_diff_unique_strings),
    ("Reassigned revoked by Citadel due to end of Campaign", reassigned_revoked_by_citadel_end_campaign),
    ("Total revoked by Citadel due to end of Campaign", total_revoked_by_citadel_end_campaign),
    ("Total Approved", total_approved),
    ("Total Revoked", total_revoked),
]

kpi_df = pd.DataFrame(kpis, columns=["Metric", "Value"])

left, right = st.columns([2, 1])

with left:
    st.markdown("#### Key Access Review Metrics")

    # Center-aligned KPI tiles (HTML/CSS) for executive readability
    st.markdown(
        f"""
        <style>
          .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
          }}
          @media (max-width: 1200px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
          }}
          .kpi-tile {{
            background: {THEME['card']};
            border: 1px solid rgba(120,120,120,0.25);
            border-radius: 12px;
            padding: 14px 12px;
            min-height: 82px;
          }}
          .kpi-label {{
            font-size: 0.85rem;
            opacity: 0.85;
            text-align: center;
            line-height: 1.1;
          }}
          .kpi-value {{
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: 6px;
            text-align: center;
            line-height: 1.1;
          }}
          .kpi-sub {{
            font-size: 0.78rem;
            opacity: 0.75;
            margin-top: 6px;
            text-align: center;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    approval_rate = (total_approved / total_identities * 100.0) if total_identities else 0.0

    tiles = [
        ("Total identities", f"{total_identities:,}", ""),
        ("Total Approved", f"{total_approved:,}", ""),
        ("Total Revoked", f"{total_revoked:,}", ""),
        ("Approval rate", f"{approval_rate:.1f}%", ""),

        ("Initial certifier count", f"{initial_certifier_count:,}", ""),
        ("Reassigned count", f"{reassigned_count:,}", ""),
        ("Unique reassigners", f"{unique_reassigners:,}", ""),
        ("Unique reassigned-to", f"{unique_reassigned_to:,}", ""),

        ("Reassigned and final actor are same", f"{reassigned_and_final_same:,}", ""),
        ("Reassigned but Final Actor is different", f"{reassigned_final_diff_count:,}", ""),
        ("Reassigned but Final Actor is different (unique strings)", f"{reassigned_final_diff_unique_strings:,}", ""),
        ("Reassigned revoked by Citadel due to end of Campaign", f"{reassigned_revoked_by_citadel_end_campaign:,}", ""),
        ("Total revoked by Citadel due to end of Campaign", f"{total_revoked_by_citadel_end_campaign:,}", ""),
    ]

    # Render tiles
    html = ['<div class="kpi-grid">']
    for label, value, sub in tiles:
        tile_html = f"""
<div class="kpi-tile">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
</div>
"""
        html.append(textwrap.dedent(tile_html).strip())
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

with right:
    st.markdown("#### Outcome split")

    total = int(total_identities)
    approved = int(total_approved)
    revoked = int(total_revoked)
    other = max(total - (approved + revoked), 0)

    pie_df = pd.DataFrame(
        {
            "Outcome": ["Approved", "Revoked"] + (["Other"] if other > 0 else []),
            "Count": [approved, revoked] + ([other] if other > 0 else []),
        }
    )

    pie = px.pie(pie_df, names="Outcome", values="Count", title="Approved vs Revoked")
    apply_plotly_theme(pie)
    pie.update_traces(textinfo="percent+value")
    pie.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(pie, use_container_width=True)
    st.caption(f"Total identities: {total:,}")

# =========================
# Reassignment Behaviour
# =========================
st.divider()
st.header("Reassignment Behaviour")

col_b = None
col_o = None
col_q = None

missing_required = [k for k in ["certifier", "reassigned_flag", "reassigned_to"] if final_map.get(k) is None]
if missing_required:
    st.error(f"Required column(s) missing: {', '.join(missing_required)}")
    st.stop()

e_true = to_bool_series_any(df[final_map["reassigned_flag"]])
d_series = df[final_map["certifier"]].fillna("").astype(str).str.strip()
f_series = df[final_map["reassigned_to"]].fillna("").astype(str).str.strip()

reassigned_mask = e_true & (f_series != "") & (d_series != f_series)
reassigned_df = df.loc[reassigned_mask].copy()

cert_col = final_map["certifier"]
flag_col = final_map["reassigned_flag"]
to_col = final_map["reassigned_to"]
actor_col = final_map.get("actor")
status_col = final_map.get("status")
updated_col = final_map.get("updated_at")
activated_col = final_map.get("activated_at")
end_col = final_map.get("end_date")
identity_col = final_map.get("identity")

total_certifications = int(len(df))
reassigned_rows = int(len(reassigned_df))
percent_reassigned = (reassigned_rows / total_certifications * 100.0) if total_certifications else 0.0
unique_reassigners = int(d_series[reassigned_mask].nunique()) if reassigned_rows else 0
unique_reassigned_to = int(f_series[reassigned_mask].nunique()) if reassigned_rows else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total certifications", f"{total_certifications:,}")
k2.metric("Reassigned rows", f"{reassigned_rows:,}")
k3.metric("Percent reassigned", f"{percent_reassigned:.1f}%")
k4.metric("Unique reassigners", f"{unique_reassigners:,}")
k5.metric("Unique reassigned-to", f"{unique_reassigned_to:,}")

if reassigned_rows == 0:
    st.info("No reassignment cases found based on reassignment flag being TRUE and reassigned-to being populated.")
else:
    st.subheader("Reassignment flow (Certifier to Reassigned to)")

    flows = (
        reassigned_df[[cert_col, to_col]]
        .assign(
            certifier=lambda x: x[cert_col].fillna("").astype(str).str.strip(),
            reassigned_to=lambda x: x[to_col].fillna("").astype(str).str.strip(),
        )[["certifier", "reassigned_to"]]
        .groupby(["certifier", "reassigned_to"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top_n_flows = 15
    top_flows = flows.head(top_n_flows).copy()
    other_flows = flows.iloc[top_n_flows:].copy()

    if len(other_flows) > 0:
        others_total = int(other_flows["count"].sum())
        top_flows = pd.concat(
            [top_flows, pd.DataFrame([{ "certifier": "Others", "reassigned_to": "Others", "count": others_total }])],
            ignore_index=True,
        )

    left_nodes = top_flows["certifier"].astype(str).unique().tolist()
    right_nodes = top_flows["reassigned_to"].astype(str).unique().tolist()
    nodes = left_nodes + [n for n in right_nodes if n not in left_nodes]
    node_index = {n: i for i, n in enumerate(nodes)}

    sankey_source = top_flows["certifier"].map(node_index)
    sankey_target = top_flows["reassigned_to"].map(node_index)
    sankey_value = top_flows["count"].astype(int)

    # --- Sankey styling: distinct colors per person (node), links inherit source color ---
    palette = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Plotly
    )

    # Assign stable colors to nodes (people). Keep "Others" neutral.
    node_colors = []
    node_color_map = {}
    color_i = 0

    for n in nodes:
        if n == "Others":
            node_color_map[n] = "rgba(160,160,160,0.9)"
        else:
            node_color_map[n] = palette[color_i % len(palette)]
            color_i += 1
        node_colors.append(node_color_map[n])

    # Links inherit color from the source certifier node (with transparency)
    def to_rgba(c, alpha=0.35):
        if isinstance(c, str) and c.startswith("rgb(") and c.endswith(")"):
            inner = c[4:-1]
            return f"rgba({inner},{alpha})"
        if isinstance(c, str) and c.startswith("rgba(") and c.endswith(")"):
            parts = c[5:-1].split(",")
            if len(parts) >= 3:
                r, g, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
                return f"rgba({r},{g},{b},{alpha})"
        # Fallback: neutral gray
        return f"rgba(160,160,160,{alpha})"

    link_colors = [to_rgba(node_colors[s], 0.35) for s in sankey_source.tolist()]

    fig_sankey = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=nodes,
                    pad=12,
                    thickness=16,
                    color=node_colors,
                    line=dict(color="rgba(80,80,80,0.6)", width=0.5),
                ),
                link=dict(
                    source=sankey_source.tolist(),
                    target=sankey_target.tolist(),
                    value=sankey_value.tolist(),
                    color=link_colors,
                ),
            )
        ]
    )
    fig_sankey.update_layout(title_text="Top reassignment flows", height=520)
    apply_plotly_theme(fig_sankey)
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.subheader("Reassignment summary table")
    flows_table = (
        flows.rename(
            columns={
                "certifier": "Certifier",
                "reassigned_to": "Reassigned to",
                "count": "Count",
            }
        )
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(flows_table, use_container_width=True, hide_index=True, height=360)

    st.subheader("Reassignment completion path")
    if actor_col is None:
        st.info("Actor column not found in this file, so reassignment completion path is unavailable.")
    else:
        path_base = (
            reassigned_df[[cert_col, to_col, actor_col]]
            .assign(
                certifier=lambda x: x[cert_col].fillna("Unknown certifier").astype(str).str.strip(),
                reassigned_to=lambda x: x[to_col].fillna("").astype(str).str.strip(),
                final_actor=lambda x: x[actor_col].fillna("").astype(str).str.strip(),
            )[["certifier", "reassigned_to", "final_actor"]]
        )
        path_base = path_base[path_base["reassigned_to"] != ""].copy()
        path_base["final_actor"] = path_base["final_actor"].replace("", "Unknown / No actor")

        if path_base.empty:
            st.info("No valid reassigned rows were found for completion path visualization.")
        else:
            layer_options = [
                "2 layers: Certifier -> Reassigned to",
                "3 layers: Certifier -> Reassigned to -> Final actor",
            ]
            if status_col is not None:
                layer_options.append("4 layers: Certifier -> Reassigned to -> Final actor -> Decision")

            selected_layers = st.radio(
                "Path layers",
                options=layer_options,
                index=2 if status_col is not None else 1,
                horizontal=True,
                key="reassign_path_layers",
            )
            include_actor = not selected_layers.startswith("2 layers")
            include_decision = selected_layers.startswith("4 layers")

            path_df = path_base.copy()
            if include_decision:
                decision_series = (
                    reassigned_df[status_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
                path_df["decision"] = decision_series
                path_df["decision"] = path_df["decision"].replace("", "Unknown")
                path_df.loc[path_df["decision"].str.lower() == "approve", "decision"] = "Approve"
                path_df.loc[path_df["decision"].str.lower() == "revoke", "decision"] = "Revoke"

            done_by_reassigned = int((path_df["final_actor"] == path_df["reassigned_to"]).sum())
            done_by_other = int(len(path_df) - done_by_reassigned)
            pct_done_by_reassigned = (done_by_reassigned / len(path_df) * 100.0) if len(path_df) else 0.0
            pct_done_by_other = (done_by_other / len(path_df) * 100.0) if len(path_df) else 0.0

            p1, p2, p3 = st.columns(3)
            p1.metric("Reassigned rows (for path)", f"{len(path_df):,}")
            p2.metric("Final actor = reassigned to", f"{done_by_reassigned:,}", f"{pct_done_by_reassigned:.1f}%")
            p3.metric("Final actor is someone else", f"{done_by_other:,}", f"{pct_done_by_other:.1f}%")

            links_lm = (
                path_df.groupby(["certifier", "reassigned_to"])
                .size()
                .reset_index(name="count")
            )
            links_mr = (
                path_df.groupby(["reassigned_to", "final_actor"])
                .size()
                .reset_index(name="count")
            )

            left_names = links_lm["certifier"].astype(str).unique().tolist()
            mid_names = (
                pd.concat([links_lm["reassigned_to"], links_mr["reassigned_to"]], ignore_index=True)
                .astype(str)
                .dropna()
                .unique()
                .tolist()
            )
            actor_names = links_mr["final_actor"].astype(str).unique().tolist()

            node_ids = []
            label_by_id = {}

            for n in left_names:
                nid = f"C::{n}"
                node_ids.append(nid)
                label_by_id[nid] = n
            for n in mid_names:
                nid = f"R::{n}"
                if nid not in node_ids:
                    node_ids.append(nid)
                label_by_id[nid] = n
            if include_actor:
                for n in actor_names:
                    nid = f"A::{n}"
                    if nid not in node_ids:
                        node_ids.append(nid)
                    label_by_id[nid] = n

            decision_names = []
            links_ad = pd.DataFrame(columns=["final_actor", "decision", "count"])
            if include_decision:
                links_ad = (
                    path_df.groupby(["final_actor", "decision"])
                    .size()
                    .reset_index(name="count")
                )
                decision_names = links_ad["decision"].astype(str).unique().tolist()
                for n in decision_names:
                    nid = f"D::{n}"
                    if nid not in node_ids:
                        node_ids.append(nid)
                    label_by_id[nid] = n

            node_index = {n: i for i, n in enumerate(node_ids)}
            node_labels = [label_by_id[nid] for nid in node_ids]
            reassigned_set = set(mid_names)

            def actor_color(actor_name: str) -> str:
                if actor_name == "Super Admin":
                    return "rgba(244,63,94,0.92)"
                if actor_name in reassigned_set:
                    return "rgba(34,197,94,0.92)"
                return "rgba(168,85,247,0.92)"

            def decision_color(decision_name: str) -> str:
                if decision_name == "Approve":
                    return "rgba(34,197,94,0.95)"
                if decision_name == "Revoke":
                    return "rgba(239,68,68,0.95)"
                return "rgba(148,163,184,0.95)"

            node_colors = []
            for nid in node_ids:
                if nid.startswith("C::"):
                    node_colors.append("rgba(59,130,246,0.90)")
                elif nid.startswith("R::"):
                    node_colors.append("rgba(245,158,11,0.90)")
                elif nid.startswith("A::"):
                    node_colors.append(actor_color(label_by_id[nid]))
                else:
                    node_colors.append(decision_color(label_by_id[nid]))

            src = []
            tgt = []
            val = []
            lnk_color = []

            for row in links_lm.itertuples(index=False):
                s_id = f"C::{row.certifier}"
                t_id = f"R::{row.reassigned_to}"
                src.append(node_index[s_id])
                tgt.append(node_index[t_id])
                val.append(int(row.count))
                lnk_color.append("rgba(59,130,246,0.26)")

            if include_actor:
                for row in links_mr.itertuples(index=False):
                    s_id = f"R::{row.reassigned_to}"
                    t_id = f"A::{row.final_actor}"
                    src.append(node_index[s_id])
                    tgt.append(node_index[t_id])
                    val.append(int(row.count))
                    if row.final_actor == "Super Admin":
                        lnk_color.append("rgba(244,63,94,0.34)")
                    elif row.final_actor == row.reassigned_to:
                        lnk_color.append("rgba(34,197,94,0.34)")
                    else:
                        lnk_color.append("rgba(168,85,247,0.30)")

            if include_decision:
                for row in links_ad.itertuples(index=False):
                    s_id = f"A::{row.final_actor}"
                    t_id = f"D::{row.decision}"
                    src.append(node_index[s_id])
                    tgt.append(node_index[t_id])
                    val.append(int(row.count))
                    if row.decision == "Approve":
                        lnk_color.append("rgba(34,197,94,0.44)")
                    elif row.decision == "Revoke":
                        lnk_color.append("rgba(239,68,68,0.44)")
                    else:
                        lnk_color.append("rgba(148,163,184,0.36)")

            fig_path = go.Figure(
                data=[
                    go.Sankey(
                        arrangement="snap",
                        node=dict(
                            label=node_labels,
                            pad=12,
                            thickness=16,
                            color=node_colors,
                            line=dict(color="rgba(80,80,80,0.55)", width=0.6),
                        ),
                        link=dict(
                            source=src,
                            target=tgt,
                            value=val,
                            color=lnk_color,
                        ),
                    )
                ]
            )
            fig_path.update_layout(
                title_text="Reassignment completion path",
                height=700 if include_decision else 620,
                margin=dict(l=20, r=20, t=60, b=20),
            )
            apply_plotly_theme(fig_path)
            st.plotly_chart(fig_path, use_container_width=True)

            if include_decision:
                st.caption(
                    "Linear flow: Certifier -> Reassigned to -> Final actor -> Decision. "
                    "No cross-stage shortcuts are included."
                )
            elif include_actor:
                st.caption(
                    "Linear flow: Certifier -> Reassigned to -> Final actor."
                )
            else:
                st.caption(
                    "Linear flow: Certifier -> Reassigned to."
                )

            with st.expander("Path table"):
                if include_decision:
                    path_table = (
                        path_df.groupby(["certifier", "reassigned_to", "final_actor", "decision"])
                        .size()
                        .reset_index(name="Count")
                        .rename(
                            columns={
                                "certifier": "Certifier",
                                "reassigned_to": "Reassigned to",
                                "final_actor": "Final actor",
                                "decision": "Decision",
                            }
                        )
                        .sort_values("Count", ascending=False)
                        .reset_index(drop=True)
                    )
                elif include_actor:
                    path_table = (
                        path_df.groupby(["certifier", "reassigned_to", "final_actor"])
                        .size()
                        .reset_index(name="Count")
                        .rename(
                            columns={
                                "certifier": "Certifier",
                                "reassigned_to": "Reassigned to",
                                "final_actor": "Final actor",
                            }
                        )
                        .sort_values("Count", ascending=False)
                        .reset_index(drop=True)
                    )
                else:
                    path_table = (
                        path_df.groupby(["certifier", "reassigned_to"])
                        .size()
                        .reset_index(name="Count")
                        .rename(
                            columns={
                                "certifier": "Certifier",
                                "reassigned_to": "Reassigned to",
                            }
                        )
                        .sort_values("Count", ascending=False)
                        .reset_index(drop=True)
                    )
                st.dataframe(path_table, use_container_width=True, hide_index=True, height=340)

    with st.expander("Cases where final actor differed from reassigned-to"):
        if actor_col is None:
            st.info("Actor column not found in this file, so final actor analysis is unavailable.")
        else:
            flows_actor = (
                reassigned_df[[cert_col, to_col, actor_col]]
                .assign(
                    certifier=lambda x: x[cert_col].fillna("").astype(str).str.strip(),
                    reassigned_to=lambda x: x[to_col].fillna("").astype(str).str.strip(),
                    actor=lambda x: x[actor_col].fillna("").astype(str).str.strip(),
                )[["certifier", "reassigned_to", "actor"]]
            )

            flows_actor = flows_actor[(flows_actor["reassigned_to"] != "") & (flows_actor["actor"] != "")]
            flows_actor = flows_actor[flows_actor["actor"] != flows_actor["reassigned_to"]]

            flows_actor_table = (
                flows_actor
                .groupby(["certifier", "reassigned_to", "actor"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .rename(
                    columns={
                        "certifier": "Certifier",
                        "reassigned_to": "Reassigned to",
                        "actor": "Final actor",
                        "count": "Count",
                    }
                )
            )

            if len(flows_actor_table) == 0:
                st.info("No cases found where a different final actor completed the action after reassignment.")
            else:
                st.dataframe(flows_actor_table, use_container_width=True, hide_index=True, height=320)

    st.subheader("Top 15 reassigned-to identities")

    # Build a stacked view: X = reassigned_to, stack = certifier, value = count
    label_threshold = 5  # show certifier labels only when segment count is >= this (avoids clutter)

    breakdown = (
        reassigned_df[[cert_col, to_col]]
        .assign(
            Certifier=lambda x: x[cert_col].fillna("NA").astype(str).str.strip(),
            Reassigned_to=lambda x: x[to_col].fillna("NA").astype(str).str.strip(),
        )
        [["Certifier", "Reassigned_to"]]
    )

    # Total per reassigned_to to pick top 15 targets
    totals = (
        breakdown.groupby("Reassigned_to")
        .size()
        .reset_index(name="Total")
        .sort_values("Total", ascending=False)
    )

    top_targets = totals.head(15)["Reassigned_to"].tolist()

    stacked = (
        breakdown[breakdown["Reassigned_to"].isin(top_targets)]
        .groupby(["Reassigned_to", "Certifier"])
        .size()
        .reset_index(name="Count")
        .merge(totals, on="Reassigned_to", how="left")
        .sort_values(["Total", "Count"], ascending=[False, False])
    )

    # Segment label: show certifier name + count only for meaningful segments
    stacked["SegLabel"] = np.where(
        stacked["Count"] >= label_threshold,
        stacked["Certifier"] + " " + stacked["Count"].astype(int).astype(str),
        "",
    )

    # Keep x-axis ordered by total descending
    x_order = totals.head(15)["Reassigned_to"].tolist()

    fig_to = px.bar(
        stacked,
        x="Reassigned_to",
        y="Count",
        color="Certifier",
        text="SegLabel",
        category_orders={"Reassigned_to": x_order},
        title="Top 15 reassigned-to identities (stacked by certifier)",
        barmode="stack",
    )

    # Make segment labels readable
    fig_to.update_traces(textposition="inside", insidetextanchor="middle")

    # Add total label on top of each bar
    totals_top = totals[totals["Reassigned_to"].isin(top_targets)].copy()
    totals_top = totals_top.set_index("Reassigned_to").loc[x_order].reset_index()

    for row in totals_top.itertuples(index=False):
        fig_to.add_annotation(
            x=row.Reassigned_to,
            y=int(row.Total),
            text=str(int(row.Total)),
            showarrow=False,
            yshift=10,
        )

    fig_to.update_layout(
        xaxis_title="Reassigned to",
        yaxis_title="Count",
        xaxis_tickangle=-45,
        height=520,
        margin=dict(l=40, r=40, t=70, b=140),
        legend_title_text="Reassigned by",
    )
    apply_plotly_theme(fig_to)
    st.plotly_chart(fig_to, use_container_width=True)

    st.caption(
        "Bars are stacked by certifier. Segment labels appear only for segments with count >= "
        f"{label_threshold}. Totals are shown on top of each bar."
    )

    st.subheader("Top 15 reassigning certifiers")
    top_reassigners = (
        reassigned_df[cert_col]
        .fillna("NA")
        .astype(str)
        .str.strip()
        .value_counts()
        .head(15)
        .reset_index()
    )
    top_reassigners.columns = ["Certifier", "Count"]
    fig_from = px.bar(top_reassigners, x="Certifier", y="Count", title="Top 15 reassigning certifiers")
    fig_from.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=40, t=60, b=120))
    apply_plotly_theme(fig_from)
    st.plotly_chart(fig_from, use_container_width=True)

    st.subheader("Reassigned-to split for top certifiers")

    # For each top certifier, show who they reassigned to (stacked).
    label_threshold = 5
    top_certifiers_n = 15
    top_targets_n = 12

    breakdown_ct = breakdown.copy()

    cert_totals = (
        breakdown_ct.groupby("Certifier")
        .size()
        .reset_index(name="Total")
        .sort_values("Total", ascending=False)
    )
    top_certifiers = cert_totals.head(top_certifiers_n)["Certifier"].tolist()

    target_totals = (
        breakdown_ct[breakdown_ct["Certifier"].isin(top_certifiers)]
        .groupby("Reassigned_to")
        .size()
        .reset_index(name="Total")
        .sort_values("Total", ascending=False)
    )
    top_targets = target_totals.head(top_targets_n)["Reassigned_to"].tolist()

    stacked_ct = breakdown_ct[breakdown_ct["Certifier"].isin(top_certifiers)].copy()
    stacked_ct["Reassigned_to"] = stacked_ct["Reassigned_to"].where(
        stacked_ct["Reassigned_to"].isin(top_targets),
        "Others",
    )

    stacked_ct = (
        stacked_ct.groupby(["Certifier", "Reassigned_to"])
        .size()
        .reset_index(name="Count")
        .merge(cert_totals, on="Certifier", how="left")
        .sort_values(["Total", "Count"], ascending=[False, False])
    )

    stacked_ct["SegLabel"] = np.where(
        stacked_ct["Count"] >= label_threshold,
        stacked_ct["Reassigned_to"] + " " + stacked_ct["Count"].astype(int).astype(str),
        "",
    )

    x_order = cert_totals.head(top_certifiers_n)["Certifier"].tolist()

    fig_ct = px.bar(
        stacked_ct,
        x="Certifier",
        y="Count",
        color="Reassigned_to",
        text="SegLabel",
        category_orders={"Certifier": x_order},
        title="Top certifiers stacked by reassigned-to recipients",
        barmode="stack",
    )

    fig_ct.update_traces(textposition="inside", insidetextanchor="middle")

    totals_top = cert_totals[cert_totals["Certifier"].isin(top_certifiers)].copy()
    totals_top = totals_top.set_index("Certifier").loc[x_order].reset_index()

    for row in totals_top.itertuples(index=False):
        fig_ct.add_annotation(
            x=row.Certifier,
            y=int(row.Total),
            text=str(int(row.Total)),
            showarrow=False,
            yshift=10,
        )

    fig_ct.update_layout(
        xaxis_title="Certifier",
        yaxis_title="Count",
        xaxis_tickangle=-45,
        height=560,
        margin=dict(l=40, r=40, t=70, b=160),
        legend_title_text="Reassigned to",
    )
    apply_plotly_theme(fig_ct)
    st.plotly_chart(fig_ct, use_container_width=True)

    st.caption(
        "Each bar is a certifier. Each coloured segment shows who they reassigned to. "
        f"Segment labels show only when count is at least {label_threshold}. Totals are shown on top."
    )

    with st.expander("Reassignment drilldown (rows)"):
        mapped_cols = [
            identity_col, cert_col, flag_col, to_col, actor_col, status_col, updated_col, activated_col, end_col
        ]
        cols_to_show = [c for c in mapped_cols if c is not None and c in reassigned_df.columns]
        show_df = reassigned_df[cols_to_show].copy() if cols_to_show else reassigned_df.copy()
        st.dataframe(show_df, use_container_width=True, height=420)

# =========================
# Outcome trend (daily approvals vs revocations)
# =========================
st.divider()
st.header("Outcome trend")

status_key = final_map.get("status")
updated_key = final_map.get("updated_at")
activated_key = final_map.get("activated_at")
end_key = final_map.get("end_date")

if status_key is None:
    st.info("Status column not found, so approval vs revocation trend cannot be computed.")
elif updated_key is None:
    st.info("Updated at column not found, so action-date trend cannot be computed.")
elif activated_key is None:
    st.info("Activated at column not found, so campaign start date cannot be computed.")
elif end_key is None:
    st.info("End date column not found, so campaign end date cannot be computed.")
else:
    activated_dt = pd.to_datetime(df[activated_key], errors="coerce", dayfirst=True)
    end_dt = pd.to_datetime(df[end_key], errors="coerce", dayfirst=True)
    action_dt = pd.to_datetime(df[updated_key], errors="coerce", dayfirst=True)

    if not activated_dt.notna().any():
        st.info("Activated at column has no usable values.")
        st.stop()
    if not end_dt.notna().any():
        st.info("End date column has no usable values.")
        st.stop()

    campaign_start = activated_dt.dt.date.dropna().min()
    campaign_end = end_dt.dt.date.dropna().max()

    if campaign_end < campaign_start:
        st.info("Campaign end date appears earlier than start date. Please verify activated and end date columns.")
        st.stop()

    status_series = df[status_key].fillna("").astype(str).str.strip()
    action_day = action_dt.dt.date

    trend_df = pd.DataFrame({"day": action_day, "status": status_series})
    trend_df = trend_df[trend_df["status"].isin(["Approve", "Revoke"]) & trend_df["day"].notna()]

    if trend_df.empty:
        st.info("No Approve or Revoke rows with a valid updated_at date were found.")
    else:
        counts = trend_df.groupby(["day", "status"]).size().reset_index(name="count")

        pivot = (
            counts.pivot(index="day", columns="status", values="count")
            .fillna(0)
            .astype(int)
            .sort_index()
        )

        if "Approve" not in pivot.columns:
            pivot["Approve"] = 0
        if "Revoke" not in pivot.columns:
            pivot["Revoke"] = 0

        all_days = pd.date_range(pd.to_datetime(campaign_start), pd.to_datetime(campaign_end), freq="D")
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.reindex(all_days, fill_value=0)

        st.caption(
            f"Campaign window: {campaign_start.strftime('%d %b %Y')} to {campaign_end.strftime('%d %b %Y')}. "
            "Daily action counts based on updated_at."
        )

        # 15-day window scroll
        x_vals = pivot.index
        window = 15
        max_start = max(len(x_vals) - window, 0)

        start_idx = 0
        if max_start > 0:
            start_idx = st.slider(
                "Scroll window",
                min_value=0,
                max_value=max_start,
                value=0,
                step=1,
            )

        end_idx = min(start_idx + window, len(x_vals))
        view_x = x_vals[start_idx:end_idx]
        view_approve = pivot["Approve"].iloc[start_idx:end_idx]
        view_revoke = pivot["Revoke"].iloc[start_idx:end_idx]

        view_start = view_x[0].date() if len(view_x) else campaign_start
        view_end = view_x[-1].date() if len(view_x) else campaign_end

        legend_suffix = f" ({view_start.strftime('%d %b')} to {view_end.strftime('%d %b')})"

        # Labels on EVERY point (hide zeros to keep it readable)
        def labels_for(series):
            vals = series.astype(int).tolist()
            return [str(v) if v != 0 else "" for v in vals]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=view_x,
                y=view_approve.values,
                mode="lines+markers+text",
                name=f"Approved{legend_suffix}",
                text=labels_for(view_approve),
                textposition="top center",
                line=dict(color="green"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=view_x,
                y=view_revoke.values,
                mode="lines+markers+text",
                name=f"Revoked{legend_suffix}",
                text=labels_for(view_revoke),
                textposition="top center",
                line=dict(color="red"),
                )
        )

        fig.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis=dict(title="Date", tickformat="%d %b"),
            yaxis=dict(title="Count"),
            title="Daily outcomes with labels on every point",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_plotly_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Trend data (table)"):
            table = pd.DataFrame(
                {
                    "Date": [d.strftime("%d %b") for d in view_x],
                    "Approve": view_approve.astype(int).values,
                    "Revoke": view_revoke.astype(int).values,
                }
            )
            st.dataframe(table, use_container_width=True, hide_index=True, height=320)

# =========================
# Compliance map (animated daily playback)
# =========================
st.divider()
st.header("Compliance map")

comp_status_key = final_map.get("status")
comp_updated_key = final_map.get("updated_at")
comp_certifier_key = final_map.get("certifier")
comp_actor_key = final_map.get("actor")
comp_flag_key = final_map.get("reassigned_flag")
comp_activated_key = final_map.get("activated_at")
comp_end_key = final_map.get("end_date")

comp_missing = [k for k in ["status", "updated_at"] if final_map.get(k) is None]
if comp_missing:
    st.info(
        "Compliance map needs status and updated_at columns. "
        f"Missing: {', '.join(comp_missing)}"
    )
else:
    comp_status = df[comp_status_key].fillna("").astype(str).str.strip()
    comp_updated_dt = pd.to_datetime(df[comp_updated_key], errors="coerce", dayfirst=True)
    comp_day = comp_updated_dt.dt.floor("D")

    comp_daily_df = pd.DataFrame({"day": comp_day, "status": comp_status})
    comp_daily_df = comp_daily_df[
        comp_daily_df["status"].isin(["Approve", "Revoke"]) & comp_daily_df["day"].notna()
    ].copy()

    if comp_daily_df.empty:
        st.info("No daily outcome rows found with valid updated_at values for Compliance map.")
    else:
        day_status_counts = (
            comp_daily_df
            .groupby(["day", "status"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        if "Approve" not in day_status_counts.columns:
            day_status_counts["Approve"] = 0
        if "Revoke" not in day_status_counts.columns:
            day_status_counts["Revoke"] = 0

        range_start = day_status_counts.index.min()
        range_end = day_status_counts.index.max()

        # Prefer campaign window when available, otherwise fallback to action-date window.
        if comp_activated_key is not None and comp_end_key is not None:
            comp_activated_dt = pd.to_datetime(df[comp_activated_key], errors="coerce", dayfirst=True)
            comp_end_dt = pd.to_datetime(df[comp_end_key], errors="coerce", dayfirst=True)
            if comp_activated_dt.notna().any() and comp_end_dt.notna().any():
                window_start = comp_activated_dt.dt.floor("D").dropna().min()
                window_end = comp_end_dt.dt.floor("D").dropna().max()
                if window_end >= window_start:
                    range_start = window_start
                    range_end = window_end

        all_days = pd.date_range(range_start, range_end, freq="D")
        day_status_counts = day_status_counts.reindex(all_days, fill_value=0)

        compliance_daily = pd.DataFrame(index=all_days)
        compliance_daily["Approved"] = day_status_counts["Approve"].astype(int)
        compliance_daily["Revoked"] = day_status_counts["Revoke"].astype(int)
        compliance_daily["Action total"] = (
            compliance_daily["Approved"] + compliance_daily["Revoked"]
        ).astype(int)

        if comp_flag_key is not None:
            comp_reassigned = to_bool_series_any(df[comp_flag_key])
            reassigned_day_counts = (
                pd.DataFrame({"day": comp_day[comp_reassigned]})
                .dropna()
                .groupby("day")
                .size()
                .reindex(all_days, fill_value=0)
                .astype(int)
            )
            compliance_daily["Reassigned"] = reassigned_day_counts.values

        if comp_certifier_key is not None:
            certifier_clean = df[comp_certifier_key].fillna("").astype(str).str.strip()
            certifier_day = (
                pd.DataFrame({"day": comp_day, "certifier": certifier_clean})
                .dropna(subset=["day"])
            )
            certifier_day = certifier_day[certifier_day["certifier"] != ""]
            certifier_counts = (
                certifier_day.groupby("day")["certifier"]
                .nunique()
                .reindex(all_days, fill_value=0)
                .astype(int)
            )
            compliance_daily["Active certifiers"] = certifier_counts.values

        if comp_actor_key is not None:
            actor_clean = df[comp_actor_key].fillna("").astype(str).str.strip()
            actor_day = pd.DataFrame({"day": comp_day, "actor": actor_clean}).dropna(subset=["day"])
            actor_day = actor_day[actor_day["actor"] != ""]
            actor_counts = (
                actor_day.groupby("day")["actor"]
                .nunique()
                .reindex(all_days, fill_value=0)
                .astype(int)
            )
            compliance_daily["Unique actors"] = actor_counts.values

        preferred_metric_order = [
            "Action total",
            "Approved",
            "Revoked",
            "Reassigned",
            "Active certifiers",
            "Unique actors",
        ]
        metric_order = [m for m in preferred_metric_order if m in compliance_daily.columns]

        long_daily = (
            compliance_daily.reset_index(names="day")
            .melt(id_vars="day", value_vars=metric_order, var_name="Metric", value_name="Count")
            .sort_values(["day", "Metric"])
        )
        long_daily["Metric"] = pd.Categorical(long_daily["Metric"], categories=metric_order, ordered=True)
        long_daily["DayLabel"] = long_daily["day"].dt.strftime("%d %b %Y")
        long_daily["Bubble"] = long_daily["Count"].clip(lower=1)
        day_list = sorted(long_daily["day"].dropna().unique().tolist())
        day_index = {d: i for i, d in enumerate(day_list)}

        # Build animated trail frames so older values stay visible (faint), while the current day stays bright.
        trail_frames = []
        for day in day_list:
            current_idx = day_index[day]
            frame = long_daily[long_daily["day"] <= day].copy()
            frame["FrameLabel"] = pd.to_datetime(day).strftime("%d %b %Y")
            frame["State"] = np.where(frame["day"] == day, "Current", "Historical")
            frame["Age"] = current_idx - frame["day"].map(day_index)
            frame["Bubble"] = np.where(
                frame["State"] == "Current",
                frame["Count"].clip(lower=1),
                np.maximum(1, (frame["Count"] * 0.42).astype(int)),
            )
            frame["PointId"] = (
                frame["Metric"].astype(str) + "_" + frame["day"].dt.strftime("%Y-%m-%d")
            )
            frame["ValueLabel"] = np.where(
                frame["State"] == "Current",
                frame["Count"].astype(int).astype(str),
                "",
            )
            trail_frames.append(frame)

        consolidated = (
            long_daily.groupby("Metric", as_index=False)["Count"]
            .sum()
            .rename(columns={"Count": "Count"})
        )
        consolidated["day"] = pd.NaT
        consolidated["DayLabel"] = "Final consolidated"
        consolidated["FrameLabel"] = "Final consolidated"
        consolidated["State"] = "Final"
        consolidated["Age"] = 0
        consolidated["Bubble"] = consolidated["Count"].clip(lower=1)
        consolidated["PointId"] = consolidated["Metric"].astype(str) + "_final"
        consolidated["ValueLabel"] = consolidated["Count"].astype(int).astype(str)

        trail_daily = pd.concat(trail_frames + [consolidated], ignore_index=True)
        long_daily_with_final = pd.concat(
            [
                long_daily.assign(FrameLabel=long_daily["DayLabel"]),
                consolidated.copy(),
            ],
            ignore_index=True,
        )

        max_count = int(max(trail_daily["Count"].max(), 1))
        color_seq = ["#2dd4bf", "#22c55e", "#ef4444", "#f59e0b", "#3b82f6", "#a855f7"]
        controls_left, controls_right = st.columns([3, 2])
        with controls_left:
            map_style = st.selectbox(
                "Compliance map style",
                ["Pulse Lanes", "Radar Sweep", "Command Bars"],
                index=0,
                help="Switch between animated compliance map views.",
            )
        with controls_right:
            frame_ms = st.slider(
                "Playback speed (ms/frame)",
                min_value=250,
                max_value=1500,
                value=650,
                step=50,
            )

        if map_style == "Pulse Lanes":
            fig_map = px.scatter(
                trail_daily,
                x="Count",
                y="Metric",
                animation_frame="FrameLabel",
                animation_group="PointId",
                size="Bubble",
                color="Metric",
                symbol="State",
                size_max=62,
                range_x=[0, max_count * 1.15 + 1],
                category_orders={"Metric": metric_order},
                color_discrete_sequence=color_seq,
                hover_data={
                    "Count": True,
                    "Metric": True,
                    "FrameLabel": True,
                    "State": True,
                    "day": False,
                    "Bubble": False,
                    "Age": False,
                    "PointId": False,
                },
                title="Compliance map: Pulse Lanes",
            )
            fig_map.update_traces(
                mode="markers+text",
                text=trail_daily["ValueLabel"],
                textposition="middle center",
                marker=dict(line=dict(width=1.5, color="rgba(255,255,255,0.40)"), opacity=0.9),
            )
            for tr in fig_map.data:
                if "Historical" in str(tr.name):
                    tr.update(opacity=0.18, marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.18)")))
                elif "Current" in str(tr.name):
                    tr.update(opacity=0.95, marker=dict(line=dict(width=1.8, color="rgba(255,255,255,0.50)")))
                elif "Final" in str(tr.name):
                    tr.update(opacity=1.0, marker=dict(line=dict(width=2.2, color="#FFE066")))
            fig_map.update_layout(
                height=520,
                margin=dict(l=40, r=20, t=70, b=50),
                xaxis_title="Daily count intensity",
                yaxis_title="Metric lane",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_map.update_yaxes(categoryorder="array", categoryarray=metric_order[::-1])
            fig_map.update_xaxes(showgrid=True, zeroline=False)
            caption_text = (
                "Historical values remain as faint traces, current-day values are bright, and the final frame shows consolidated totals."
            )
        elif map_style == "Radar Sweep":
            radar_daily = long_daily_with_final.copy()
            fig_map = px.line_polar(
                radar_daily,
                r="Count",
                theta="Metric",
                animation_frame="FrameLabel",
                line_close=True,
                category_orders={"Metric": metric_order},
                title="Compliance map: Radar Sweep",
            )
            fig_map.update_traces(
                fill="toself",
                line=dict(width=3, color="#00E5A8"),
                marker=dict(size=8, color="#00E5A8"),
                opacity=0.55,
            )
            fig_map.update_layout(
                height=560,
                margin=dict(l=30, r=30, t=70, b=20),
                showlegend=False,
                polar=dict(
                    radialaxis=dict(range=[0, max_count * 1.1 + 1], showticklabels=True, ticks="outside"),
                ),
            )
            caption_text = (
                "Radar sweep animates each day and ends with a consolidated footprint in the final frame."
            )
        else:
            bars_daily = long_daily_with_final.copy()
            fig_map = px.bar(
                bars_daily,
                x="Metric",
                y="Count",
                color="Metric",
                animation_frame="FrameLabel",
                category_orders={"Metric": metric_order},
                color_discrete_sequence=color_seq,
                title="Compliance map: Command Bars",
            )
            fig_map.update_layout(
                height=520,
                margin=dict(l=40, r=20, t=70, b=50),
                xaxis_title="Compliance metric",
                yaxis_title="Daily count",
                yaxis=dict(range=[0, max_count * 1.15 + 1]),
                showlegend=False,
            )
            fig_map.update_xaxes(tickangle=-20)
            caption_text = (
                "Command bars step through daily shifts and finish with consolidated totals in the final stage."
            )

        apply_plotly_theme(fig_map)

        # Tune animation speed for play button.
        if fig_map.layout.updatemenus:
            for menu in fig_map.layout.updatemenus:
                if menu.buttons and len(menu.buttons) > 0:
                    menu.buttons[0].args[1]["frame"]["duration"] = frame_ms
                    menu.buttons[0].args[1]["transition"]["duration"] = int(frame_ms * 0.6)

        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(caption_text)

        heat = compliance_daily[metric_order].T.copy()
        heat.columns = [d.strftime("%d %b") for d in heat.columns]

        fig_heat = px.imshow(
            heat,
            aspect="auto",
            labels=dict(x="Day", y="Metric", color="Count"),
            color_continuous_scale="Turbo",
            title="Compliance map overview (all days)",
        )
        fig_heat.update_layout(height=320, margin=dict(l=40, r=20, t=55, b=30))
        apply_plotly_theme(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("Compliance map data (table)"):
            show_comp = compliance_daily.reset_index(names="Date")
            show_comp["Date"] = show_comp["Date"].dt.strftime("%d %b %Y")
            st.dataframe(show_comp, use_container_width=True, hide_index=True, height=320)

# =========================
# Certifier statistics (table)
# =========================
st.divider()
st.header("Certifier statistics")

cert_key = final_map.get("certifier")
flag_key = final_map.get("reassigned_flag")
to_key = final_map.get("reassigned_to")
actor_key = final_map.get("actor")
status_key = final_map.get("status")

missing = [k for k in ["certifier", "reassigned_flag", "reassigned_to", "actor", "status"] if final_map.get(k) is None]
if missing:
    st.info(f"Cannot compute certifier statistics. Missing column(s): {', '.join(missing)}")
else:
    d = df[cert_key].fillna("").astype(str).str.strip()
    e_true = to_bool_series_any(df[flag_key])
    f = df[to_key].fillna("").astype(str).str.strip()
    g = df[actor_key].fillna("").astype(str).str.strip()
    l = df[status_key].fillna("").astype(str).str.strip()

    # Normalize status defensively
    l_norm = l.str.strip()
    approve_mask = l_norm.eq("Approve")
    revoke_mask = l_norm.eq("Revoke")

    # Universe of names should include initial certifiers, reassigned recipients, and final actors
    all_names = pd.concat(
        [
            d[d.ne("")],
            f[f.ne("")],
            g[g.ne("")],
        ],
        ignore_index=True,
    ).dropna()
    certifiers = sorted(all_names.unique().tolist())

    # Reassignment masks should respect the flag
    reassigned_mask = e_true & f.ne("")

    # Did certifier reassign to someone
    out_counts = (
        pd.DataFrame({"certifier": d[reassigned_mask]})
        .groupby("certifier")
        .size()
        .rename("Did Certifier reassign to someone? count")
    )

    # Was certifier reassigned something by someone
    in_counts = (
        pd.DataFrame({"certifier": f[reassigned_mask]})
        .groupby("certifier")
        .size()
        .rename("Was Certifier reassigned something? Reassigned count")
    )

    # Approval count where certifier is final actor
    approve_actor_counts = (
        pd.DataFrame({"certifier": g[g.ne("") & approve_mask]})
        .groupby("certifier")
        .size()
        .rename("Approval count where Certifier is actor")
    )

    # Revoke count where certifier is final actor
    revoke_actor_counts = (
        pd.DataFrame({"certifier": g[g.ne("") & revoke_mask]})
        .groupby("certifier")
        .size()
        .rename("Revoke count where Certifier is actor")
    )

    # Assemble
    stats = pd.DataFrame({"Certifier name": certifiers}).set_index("Certifier name")
    stats = stats.join(out_counts, how="left")
    stats = stats.join(in_counts, how="left")
    stats = stats.join(approve_actor_counts, how="left")
    stats = stats.join(revoke_actor_counts, how="left")
    stats = stats.fillna(0).astype(int)

    # Total should only be final actions
    stats["Total count"] = stats["Approval count where Certifier is actor"] + stats["Revoke count where Certifier is actor"]

    stats = stats.reset_index().sort_values("Total count", ascending=False).reset_index(drop=True)

    # Totals row
    totals_row = {
        "Certifier name": "TOTAL",
        "Did Certifier reassign to someone? count": int(stats["Did Certifier reassign to someone? count"].sum()),
        "Was Certifier reassigned something? Reassigned count": int(stats["Was Certifier reassigned something? Reassigned count"].sum()),
        "Approval count where Certifier is actor": int(stats["Approval count where Certifier is actor"].sum()),
        "Revoke count where Certifier is actor": int(stats["Revoke count where Certifier is actor"].sum()),
        "Total count": int(stats["Total count"].sum()),
    }

    stats = pd.concat([stats, pd.DataFrame([totals_row])], ignore_index=True)

    # Serial number column
    stats.insert(0, "S No", "")
    non_total_mask = stats["Certifier name"].astype(str).ne("TOTAL")
    stats.loc[non_total_mask, "S No"] = range(1, int(non_total_mask.sum()) + 1)

    # Optional sanity check to spot mismatches early
    total_outcomes = int((approve_mask | revoke_mask).sum())
    total_actions_with_actor = int(((approve_mask | revoke_mask) & g.ne("")).sum())
    st.caption(
        f"Sanity check. Total outcomes rows: {total_outcomes:,}. "
        f"Rows with outcome and non empty actor: {total_actions_with_actor:,}. "
        f"TOTAL final actions in table: {int(totals_row['Total count']):,}."
    )

    st.dataframe(stats, use_container_width=True, hide_index=True, height=420)

# =========================
# Certifier + reassigned statistics (effective certifier)
# =========================
st.divider()
st.header("certifier+reassigned statistics")

cert_key = final_map.get("certifier")
flag_key = final_map.get("reassigned_flag")
to_key = final_map.get("reassigned_to")
status_key = final_map.get("status")

missing = [k for k in ["certifier", "reassigned_flag", "reassigned_to", "status"] if final_map.get(k) is None]
if missing:
    st.info(f"Cannot compute certifier+reassigned statistics. Missing column(s): {', '.join(missing)}")
else:
    d = df[cert_key].fillna("").astype(str).str.strip()
    e_true = to_bool_series_any(df[flag_key])
    f = df[to_key].fillna("").astype(str).str.strip()
    l = df[status_key].fillna("").astype(str).str.strip()

    # If reassigned is TRUE, replace certifier with reassigned-to identity (when present)
    merged = d.copy()
    replace_mask = e_true & (f != "")
    merged.loc[replace_mask] = f.loc[replace_mask]

    merged_df = pd.DataFrame(
        {
            "Certifier name": d,
            "Reassigned value": e_true,
            "Reassigned name": f,
            "Merged column": merged,
            "Status": l,
        }
    )

    # Only consider explicit outcomes
    merged_df["Reassigned name"] = merged_df["Reassigned name"].fillna("").astype(str).str.strip()
    merged_df = merged_df[merged_df["Status"].isin(["Approve", "Revoke"])].copy()

    if merged_df.empty:
        st.info("No Approve or Revoke rows found for certifier+reassigned statistics.")
    else:
        summary = (
            merged_df
            .groupby(["Certifier name", "Reassigned value", "Reassigned name", "Merged column", "Status"])  # preserves requested columns
            .size()
            .reset_index(name="Count")
        )

        pivot = (
            summary
            .pivot_table(
                index=["Certifier name", "Reassigned value", "Reassigned name", "Merged column"],
                columns="Status",
                values="Count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        # Ensure both columns exist
        if "Approve" not in pivot.columns:
            pivot["Approve"] = 0
        if "Revoke" not in pivot.columns:
            pivot["Revoke"] = 0

        pivot = pivot.rename(
            columns={
                "Approve": "Approved count",
                "Revoke": "Revocation count",
            }
        )

        pivot["Total"] = pivot["Approved count"].astype(int) + pivot["Revocation count"].astype(int)

        # Add serial number as left-most column (exclude TOTAL row)
        pivot.insert(0, "S No", range(1, len(pivot) + 1))

        # Executive-friendly ordering
        pivot = pivot.sort_values(["Total", "Merged column", "Reassigned name"], ascending=[False, True, True]).reset_index(drop=True)

        # Recompute serial numbers after sort
        pivot["S No"] = range(1, len(pivot) + 1)

        # Add TOTAL row
        total_row = {
            "S No": "",
            "Certifier name": "TOTAL",
            "Reassigned value": "",
            "Reassigned name": "",
            "Merged column": "",
            "Approved count": int(pivot["Approved count"].sum()),
            "Revocation count": int(pivot["Revocation count"].sum()),
            "Total": int(pivot["Total"].sum()),
        }

        pivot = pd.concat([pivot, pd.DataFrame([total_row])], ignore_index=True)

        st.dataframe(pivot, use_container_width=True, hide_index=True, height=420)

        # -------------------------
        # Unique merged-name summary (one row per effective certifier)
        # -------------------------
        st.subheader("certifier+reassigned summary (unique merged names)")

        uniq = merged_df.copy()

        # Aggregate by effective certifier name (Merged column)
        uniq_counts = (
            uniq.groupby(["Merged column", "Status"])
            .size()
            .reset_index(name="Count")
        )

        uniq_pivot = (
            uniq_counts.pivot_table(
                index=["Merged column"],
                columns="Status",
                values="Count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        if "Approve" not in uniq_pivot.columns:
            uniq_pivot["Approve"] = 0
        if "Revoke" not in uniq_pivot.columns:
            uniq_pivot["Revoke"] = 0

        uniq_pivot = uniq_pivot.rename(
            columns={
                "Merged column": "Effective certifier",
                "Approve": "Approved count",
                "Revoke": "Revocation count",
            }
        )

        uniq_pivot["Total"] = uniq_pivot["Approved count"].astype(int) + uniq_pivot["Revocation count"].astype(int)

        # Sort and add serial number
        uniq_pivot = uniq_pivot.sort_values(["Total", "Effective certifier"], ascending=[False, True]).reset_index(drop=True)
        uniq_pivot.insert(0, "S No", range(1, len(uniq_pivot) + 1))

        # TOTAL row
        total_row2 = {
            "S No": "",
            "Effective certifier": "TOTAL",
            "Approved count": int(uniq_pivot["Approved count"].sum()),
            "Revocation count": int(uniq_pivot["Revocation count"].sum()),
            "Total": int(uniq_pivot["Total"].sum()),
        }
        uniq_pivot = pd.concat([uniq_pivot, pd.DataFrame([total_row2])], ignore_index=True)

        st.dataframe(uniq_pivot, use_container_width=True, hide_index=True, height=420)

# =========================
# Actor Distribution (at end)
# =========================
st.divider()
st.header("Actor Distribution")

actor_dist_col = final_map.get("actor")
if actor_dist_col is None:
    st.info("Actor column not found in this file.")
else:
    actor_series = df[actor_dist_col].fillna("NA").astype(str).str.strip()
    actor_counts = actor_series.value_counts().reset_index()
    actor_counts.columns = ["Actor", "Count"]

    top_n = 15
    visible_df = actor_counts.head(top_n)

    bar = px.bar(visible_df, x="Actor", y="Count", title="Top 15 Actors by Count")
    apply_plotly_theme(bar)
    bar.update_layout(xaxis_tickangle=-45, height=420, margin=dict(l=40, r=40, t=60, b=120))
    st.plotly_chart(bar, use_container_width=True)

    with st.expander("View remaining actors"):
        st.dataframe(actor_counts.iloc[top_n:], use_container_width=True, hide_index=True, height=300)
