import pandas as pd
import plotly.express as px

def categorical_bar(df: pd.DataFrame, col: str, top_n: int = 15):
    s = df[col].astype(str).fillna("NA")
    top = s.value_counts().head(top_n).reset_index()
    top.columns = [col, "count"]
    return px.bar(top, x=col, y="count", title=f"Top values in {col}")