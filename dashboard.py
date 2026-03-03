import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="CDPR AI Dashboard", layout="wide")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("CDPR AI – Engineering Risk Intelligence")
st.caption("Behavioral Risk Monitoring System")
st.caption(f"Last Updated: {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
st.markdown("---")

# --------------------------------------------------
# FETCH DATA
# --------------------------------------------------
response = requests.get(f"{API_URL}/manager/alerts")

if response.status_code != 200:
    st.error("Backend API not reachable.")
    st.stop()

data = response.json()
df = pd.DataFrame(data["developers"])

if df.empty:
    st.info("No developer data available.")
    st.stop()

# Split by severity
df_high = df[df["risk_level"] == "HIGH"]
df_medium = df[df["risk_level"] == "MEDIUM"]
df_low = df[df["risk_level"] == "LOW"]

# --------------------------------------------------
# KPI ROW
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("High Risk", len(df_high))
col2.metric("Medium Risk", len(df_medium))
col3.metric("Low Risk", len(df_low))

st.markdown("---")

# --------------------------------------------------
# PIE CHART (HIGH RISK ONLY)
# --------------------------------------------------
st.subheader("High Risk Developer Distribution")

if not df_high.empty:
    fig_pie = px.pie(
        df_high,
        names="developer",
        values="risk_score",
        title="High Risk Employees Breakdown",
        color_discrete_sequence=px.colors.sequential.Reds
    )
    fig_pie.update_layout(template="simple_white")
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("No HIGH risk developers detected.")

st.markdown("---")

# --------------------------------------------------
# TABS SECTION
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["HIGH Risk", "MEDIUM Risk", "LOW Risk"])

# ================= HIGH TAB =================
with tab1:
    st.subheader("HIGH Risk Developers")

    if not df_high.empty:

        fig_bar = px.bar(
            df_high.sort_values("risk_score", ascending=False),
            x="developer",
            y="risk_score",
            color="risk_score",
            color_continuous_scale="Reds"
        )
        fig_bar.update_layout(template="simple_white")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            df_high.sort_values("risk_score", ascending=False),
            use_container_width=True
        )

        csv = df_high.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download HIGH Risk CSV",
            csv,
            "high_risk_developers.csv",
            "text/csv"
        )

    else:
        st.success("No HIGH risk developers.")

# ================= MEDIUM TAB =================
with tab2:
    st.subheader("MEDIUM Risk Developers")

    if not df_medium.empty:

        fig_bar = px.bar(
            df_medium.sort_values("risk_score", ascending=False),
            x="developer",
            y="risk_score",
            color="risk_score",
            color_continuous_scale="Oranges"
        )
        fig_bar.update_layout(template="simple_white")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            df_medium.sort_values("risk_score", ascending=False),
            use_container_width=True
        )

        csv = df_medium.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download MEDIUM Risk CSV",
            csv,
            "medium_risk_developers.csv",
            "text/csv"
        )

    else:
        st.info("No MEDIUM risk developers.")

# ================= LOW TAB =================
with tab3:
    st.subheader("LOW Risk Developers")

    if not df_low.empty:

        fig_bar = px.bar(
            df_low.sort_values("risk_score", ascending=False),
            x="developer",
            y="risk_score",
            color="risk_score",
            color_continuous_scale="Greens"
        )
        fig_bar.update_layout(template="simple_white")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            df_low.sort_values("risk_score", ascending=False),
            use_container_width=True
        )

        csv = df_low.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download LOW Risk CSV",
            csv,
            "low_risk_developers.csv",
            "text/csv"
        )

    else:
        st.info("No LOW risk developers.")