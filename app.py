"""
Streamlit app: Nenana Ice Classic — interactive scatter plot.
Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import streamlit as st


def decimal_day_to_date(decimal_day, year):
    dt = datetime(int(year), 1, 1) + timedelta(days=float(decimal_day) - 1)
    return dt.strftime("%Y-%m-%d")


def fmt(val, decimals=1, suffix=""):
    if pd.isna(val):
        return "N/A"
    if decimals == 0:
        return f"{int(val)}{suffix}"
    return f"{val:.{decimals}f}{suffix}"


@st.cache_data
def load_data():
    df = pd.read_csv("NenanaIceClassic_1917-2021.csv")
    has_breakup = df["Decimal Day of Year"].notna()
    has_march   = df["Latest March Ice Reading"].notna()
    df_plot = df[has_breakup & has_march].copy()
    df_plot["breakup_date"] = df_plot.apply(
        lambda r: decimal_day_to_date(r["Decimal Day of Year"], r["Year"]), axis=1
    )
    df_plot["decade"] = (df_plot["Year"] // 10 * 10).astype(str) + "s"
    return df_plot


DECADE_COLORS = {
    "1980s": "#4e79a7",
    "1990s": "#f28e2b",
    "2000s": "#59a14f",
    "2010s": "#e15759",
    "2020s": "#b07aa1",
}

TICK_VALS  = list(range(95, 146, 5))
TICK_TEXTS = [
    (datetime(2000, 1, 1) + timedelta(days=d - 1)).strftime("%b %d")
    for d in TICK_VALS
]


def build_figure(df, show_trend):
    fig = go.Figure()

    for decade, grp in df.groupby("decade"):
        color = DECADE_COLORS.get(decade, "#888888")

        customdata = np.stack([
            grp["breakup_date"],
            grp["Year"].astype(int).astype(str),
            grp["Month"].fillna("N/A"),
            grp["Day"].apply(lambda v: fmt(v, 0)),
            grp["Time"].fillna("N/A"),
            # March ice
            grp["Latest March Ice Reading"].apply(lambda v: fmt(v, 1, " in")),
            grp["Latest March Date"].apply(lambda v: fmt(v, 0)),
            grp["march_high_temp"].apply(lambda v: fmt(v, 0, "°F")),
            grp["march_low_temp"].apply(lambda v: fmt(v, 0, "°F")),
            grp["march_avg_temp"].apply(lambda v: fmt(v, 2, "°F")),
            # Feb ice
            grp["Latest Feb Ice Reading"].apply(lambda v: fmt(v, 1, " in")),
            grp["Latest Feb Date"].apply(lambda v: fmt(v, 0)),
            grp["feb_high_temp"].apply(lambda v: fmt(v, 0, "°F")),
            grp["feb_low_temp"].apply(lambda v: fmt(v, 0, "°F")),
            grp["feb_avg_temp"].apply(lambda v: fmt(v, 2, "°F")),
            # Climate indices
            grp["nino34_feb"].apply(lambda v: fmt(v, 2)),
            grp["nino34_march"].apply(lambda v: fmt(v, 2)),
            grp["ao_feb"].apply(lambda v: fmt(v, 3)),
            grp["ao_march"].apply(lambda v: fmt(v, 3)),
            grp["pdo_feb"].apply(lambda v: fmt(v, 2)),
            grp["pdo_march"].apply(lambda v: fmt(v, 2)),
            # Decimal day
            grp["Decimal Day of Year"].apply(lambda v: fmt(v, 4)),
        ], axis=-1)

        fig.add_trace(go.Scatter(
            x=grp["Latest March Ice Reading"],
            y=grp["Decimal Day of Year"],
            mode="markers",
            name=decade,
            marker=dict(color=color, size=10, opacity=0.85,
                        line=dict(width=0.6, color="white")),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[1]} — Breakup %{customdata[0]}</b>"
                "  (%{customdata[2]} %{customdata[3]}, %{customdata[4]})<br>"
                "<br>"
                "<b>March</b><br>"
                "  Ice thickness:  %{customdata[5]}  (reading taken day %{customdata[6]})<br>"
                "  Temperature:    High %{customdata[7]}  /  Low %{customdata[8]}  /  Avg %{customdata[9]}<br>"
                "<br>"
                "<b>February</b><br>"
                "  Ice thickness:  %{customdata[10]}  (reading taken day %{customdata[11]})<br>"
                "  Temperature:    High %{customdata[12]}  /  Low %{customdata[13]}  /  Avg %{customdata[14]}<br>"
                "<br>"
                "<b>Climate indices</b><br>"
                "  ENSO Niño 3.4:  Feb %{customdata[15]}  /  Mar %{customdata[16]}<br>"
                "  AO index:       Feb %{customdata[17]}  /  Mar %{customdata[18]}<br>"
                "  PDO index:      Feb %{customdata[19]}  /  Mar %{customdata[20]}<br>"
                "<br>"
                "Decimal day of year: %{customdata[21]}"
                "<extra></extra>"
            ),
        ))

    if show_trend:
        x = df["Latest March Ice Reading"].values
        y = df["Decimal Day of Year"].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() - 1, x.max() + 1, 200)
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines",
            name=f"Trend ({m:+.2f} days/in)",
            line=dict(color="gray", dash="dash", width=1.5),
            hoverinfo="skip",
        ))

    fig.update_layout(
        xaxis=dict(title="Latest March Ice Thickness (inches)", gridcolor="#eeeeee", zeroline=False),
        yaxis=dict(title="Breakup date", tickvals=TICK_VALS, ticktext=TICK_TEXTS,
                   gridcolor="#eeeeee", zeroline=False),
        legend=dict(title="Decade", bordercolor="#dddddd", borderwidth=1),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=13, align="left"),
        margin=dict(l=60, r=20, t=20, b=60),
        hovermode="closest",
    )
    return fig


# ── Streamlit layout ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Nenana Ice Classic", layout="wide")
st.title("Nenana Ice Classic — March Ice Thickness vs. Breakup Date")
st.caption(
    "Hover over any point to see the full breakup date, ice readings, temperatures, "
    "and climate indices (ENSO / AO / PDO) for that year."
)

df = load_data()

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    year_range = st.slider(
        "Year range",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=(int(df["Year"].min()), int(df["Year"].max())),
    )
with col2:
    decades = sorted(df["decade"].unique())
    selected_decades = st.multiselect("Decades", decades, default=decades)
with col3:
    show_trend = st.checkbox("Show trend line", value=True)

mask = (
    df["Year"].between(*year_range) &
    df["decade"].isin(selected_decades)
)
df_filtered = df[mask]

if df_filtered.empty:
    st.warning("No data for the selected filters.")
else:
    fig = build_figure(df_filtered, show_trend)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        display_cols = [
            "Year", "breakup_date", "Decimal Day of Year",
            "Latest March Ice Reading", "march_high_temp", "march_low_temp", "march_avg_temp",
            "Latest Feb Ice Reading", "feb_high_temp", "feb_low_temp", "feb_avg_temp",
            "nino34_feb", "nino34_march", "ao_feb", "ao_march", "pdo_feb", "pdo_march",
        ]
        st.dataframe(
            df_filtered[display_cols].sort_values("Year", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )
