
# streamlit_app.py
import streamlit as st
from datetime import date, datetime
from decimal import Decimal
import pandas as pd
import plotly.express as px
import io

# ---- EMI Engine ----
from decimal import ROUND_HALF_UP, getcontext
from dateutil.relativedelta import relativedelta

getcontext().prec = 28

def to_decimal(x):
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))

def money(x):
    return to_decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def compute_emi(principal, annual_rate_percent, months):
    P = to_decimal(principal)
    n = int(months)
    r = to_decimal(annual_rate_percent) / Decimal(1200)  # monthly rate
    if n <= 0:
        raise ValueError("Months must be > 0")
    if r == 0:
        return money(P / n)
    one_plus_r_pow_n = (Decimal(1) + r) ** n
    emi = P * r * one_plus_r_pow_n / (one_plus_r_pow_n - Decimal(1))
    return money(emi)

def parse_extra_payments(s):
    # format: YYYY-MM-DD:amount,YYYY-MM-DD:amount
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        d_str, amt_str = part.split(":")
        y, m, d = map(int, d_str.split("-"))
        out[date(y, m, d)] = Decimal(amt_str)
    return out

def parse_rate_changes(s):
    # format: YYYY-MM-DD:rate,YYYY-MM-DD:rate
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        d_str, rate_str = part.split(":")
        y, m, d = map(int, d_str.split("-"))
        out.append((date(y, m, d), Decimal(rate_str)))
    return out

def build_emi_schedule(
    principal,
    annual_rate_percent,
    months,
    start_date,
    extra_payments=None,
    rate_changes=None,
    keep_emi_on_rate_change=False
):
    P = to_decimal(principal)
    r_annual = to_decimal(annual_rate_percent)
    n = int(months)
    extra_payments = extra_payments or {}
    rate_changes = sorted(rate_changes or [], key=lambda x: x[0])

    def monthly_rate(apr_percent):
        return to_decimal(apr_percent) / Decimal(1200)

    # EMI dates: first EMI one month after start_date
    emi_dates = []
    dt = start_date + relativedelta(months=+1)
    for _ in range(n + 600):  # cap
        emi_dates.append(dt)
        dt = dt + relativedelta(months=+1)

    current_rate = r_annual
    r_m = monthly_rate(current_rate)
    emi = compute_emi(P, current_rate, n)

    schedule_rows = []
    opening = P
    i = 0
    date_idx = 0
    rc_idx = 0

    while opening > Decimal("0.004") and date_idx < len(emi_dates) and i < n + 600:
        pay_date = emi_dates[date_idx]
        date_idx += 1
        i += 1

        # rate changes at or before this pay_date
        while rc_idx < len(rate_changes) and rate_changes[rc_idx][0] <= pay_date:
            current_rate = to_decimal(rate_changes[rc_idx][1])
            r_m = monthly_rate(current_rate)
            if not keep_emi_on_rate_change:
                remaining_months = max(1, n - (i - 1))
                emi = compute_emi(opening, current_rate, remaining_months)
            rc_idx += 1

        interest = money(opening * r_m)
        principal_component = emi - interest
        extra = money(extra_payments.get(pay_date, 0))

        if principal_component + extra > opening:
            principal_component = opening - extra
            if principal_component < 0:
                extra = opening
                principal_component = Decimal("0.00")
            emi_effective = interest + principal_component
        else:
            emi_effective = emi

        closing = opening - principal_component - extra
        if closing < Decimal("0.00") and closing > Decimal("-0.01"):
            closing = Decimal("0.00")

        schedule_rows.append({
            "#": i,
            "Date": pay_date,
            "Opening": float(money(opening)),
            "EMI": float(money(emi_effective)),
            "Interest": float(money(interest)),
            "Principal": float(money(principal_component)),
            "Extra": float(money(extra)),
            "Closing": float(money(closing)),
            "Rate%": float(current_rate),
        })

        opening = closing

    df = pd.DataFrame(schedule_rows)
    totals = {
        "EMI (current)": float(emi) if len(df) > 0 else None,
        "Total Interest": float(money(df["Interest"].sum())) if not df.empty else 0.0,
        "Total Principal": float(money(df["Principal"].sum() + df["Extra"].sum())) if not df.empty else 0.0,
        "Total Paid": float(money(df["EMI"].sum())) if not df.empty else 0.0,
        "Months Taken": int(len(df)),
        "Last Payment Date": str(df["Date"].iloc[-1]) if not df.empty else "-",
        "Interest % of Total": float(round((df["Interest"].sum() / df["EMI"].sum()) * 100, 2)) if not df.empty and df["EMI"].sum() else 0.0,
    }
    return df, totals

# ---- UI ----
st.set_page_config(page_title="EMI Schedule Dashboard", layout="wide")

st.title("🏦 EMI Schedule Dashboard")
st.markdown("<h4 style='text-align: center; color: gray;'>Mulagund & Co, Chartered Accountants</h4>", unsafe_allow_html=True)
st.caption("Build amortization schedules with prepayments and rate changes. Download, chart, and compare scenarios.")

with st.sidebar:
    st.header("Inputs")
    col1, col2 = st.columns(2)
    with col1:
        principal = st.number_input("Principal (₹)", value=7000000, min_value=10000, step=10000, format="%d")
        months = st.number_input("Tenure (months)", value=240, min_value=1, step=1)
    with col2:
        annual_rate = st.number_input("Annual Interest Rate (%)", value=9.0, min_value=0.0, step=0.05, format="%.2f")
        start_date = st.date_input("Disbursal Date", value=date(2025, 9, 1))

    st.subheader("Advanced")
    keep_emi = st.toggle("Keep EMI constant on rate change (tenure varies)", value=False)
    extras_text = st.text_input("Extra Payments (YYYY-MM-DD:amount,...)", "2026-03-01:50000,2026-12-01:50000")
    rates_text = st.text_input("Rate Changes (YYYY-MM-DD:new_rate%,...)", "2027-01-01:8.5")

    run_btn = st.button("Generate Schedule", type="primary")

placeholder = st.empty()

if run_btn:
    extras = parse_extra_payments(extras_text)
    changes = parse_rate_changes(rates_text)
    df, totals = build_emi_schedule(
        Decimal(principal), Decimal(annual_rate), int(months), start_date,
        extra_payments=extras, rate_changes=changes, keep_emi_on_rate_change=keep_emi
    )

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("EMI (current)", f"₹ {totals['EMI (current)']:.2f}")
    k2.metric("Total Interest", f"₹ {totals['Total Interest']:.2f}")
    k3.metric("Total Principal", f"₹ {totals['Total Principal']:.2f}")
    k4.metric("Total Paid", f"₹ {totals['Total Paid']:.2f}")
    k5.metric("Months Taken", f"{totals['Months Taken']}")
    k6.metric("Last Payment", totals["Last Payment Date"])

    st.divider()
    st.subheader("Schedule")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Charts
    st.subheader("Charts")
    c1, c2 = st.columns(2)
    with c1:
        line_df = df[["Date","Opening","Closing"]]
        fig1 = px.line(line_df, x="Date", y=["Opening","Closing"], title="Balance Over Time")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        bar_df = df[["Date","Interest","Principal","Extra"]]
        bar_df_melt = bar_df.melt(id_vars="Date", var_name="Component", value_name="Amount")
        fig2 = px.bar(bar_df_melt, x="Date", y="Amount", color="Component", title="Monthly Interest vs Principal vs Extra", barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

    # Composition pie
    st.subheader("Total Composition")
    pie_df = pd.DataFrame({
        "Component": ["Interest","Principal (incl. Extra)"],
        "Amount": [df["Interest"].sum(), (df["Principal"].sum() + df["Extra"].sum())]
    })
    fig3 = px.pie(pie_df, names="Component", values="Amount", title="Interest vs Principal")
    st.plotly_chart(fig3, use_container_width=True)

    # Downloads
    st.subheader("Download")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="emi_schedule.csv", mime="text/csv")

    # Excel
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Schedule")
        # KPI sheet
        pd.DataFrame([totals]).to_excel(writer, index=False, sheet_name="Summary")
    st.download_button("⬇️ Download Excel", towrite.getvalue(), file_name="emi_schedule.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    placeholder.info("Set your inputs in the sidebar and click **Generate Schedule**.")
