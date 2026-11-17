import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.data import load_prices
from engine.env import load_env, setting
from engine.metrics import (concentration, portfolio_returns, volatility,
                            max_drawdown, correlation, high_pairs)
from engine.report import ai_report, unverified_numbers  # CHANGED (Groq)
from engine.score import risk_score, RULE_TABLE
from engine.universe import SECTORS
from engine.wallet import (BASE_RPC, is_valid_address, wallet_holdings,
                           holdings_to_portfolio)

load_env()  # .env locally; Streamlit Cloud uses Secrets

st.set_page_config(page_title="AUREN · See the risk before it sees you",
                   layout="wide")

# ------------------------------------------------------------------ brand
GOLD, GREY, PANEL, BG = "#E3B341", "#8A8F98", "#18181B", "#0E0E10"
TEXT, GRID = "#F5F6F7", "#26262B"
ASSET_GREYS = ["#5B5F68", "#7A7F88", "#A3A7AF", "#C9CCD2", "#E1E3E6", "#44474F"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, .stApp, [data-testid="stMarkdownContainer"], button, input {
    font-family: 'Inter', sans-serif;
}
.stApp { background: #0E0E10; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
section[data-testid="stSidebar"] { background: #121215; border-right: 1px solid #26262B; }

.wordmark { font-weight: 700; letter-spacing: 0.38em; font-size: 2.1rem;
            color: #F5F6F7; line-height: 1; }
.wordmark .ring { color: #E3B341; margin-right: 0.55rem; letter-spacing: 0; }
.wordmark.small { font-size: 1.1rem; margin-bottom: 1.2rem; }
.tagline { color: #8A8F98; margin-top: 0.7rem; font-size: 1.02rem; }

.section-title { font-size: 1.05rem; font-weight: 600; color: #F5F6F7;
                 margin: 1.6rem 0 0.2rem 0; }
.section-sub { color: #8A8F98; font-size: 0.88rem; margin-bottom: 0.7rem; }

.card { background: #18181B; border: 1px solid #26262B; border-radius: 14px;
        padding: 18px 20px; height: 100%; }
.card .label { color: #8A8F98; font-size: 0.74rem; text-transform: uppercase;
               letter-spacing: 0.09em; font-weight: 600; }
.card .value { font-family: 'JetBrains Mono', monospace; font-weight: 700;
               font-size: 2rem; color: #F5F6F7; margin: 6px 0 4px 0; }
.card .value.gold { color: #E3B341; }
.card .sub { color: #8A8F98; font-size: 0.82rem; line-height: 1.35; }

.callout { border-left: 3px solid #E3B341; background: #18181B;
           border-radius: 8px; padding: 12px 16px; margin: 8px 0;
           color: #F5F6F7; font-size: 0.95rem; }
.callout.muted { border-left-color: #44474F; color: #8A8F98; }

.chip { display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
        font-family: 'JetBrains Mono', monospace; }
.chip.ok { background: rgba(227,179,65,0.14); color: #E3B341; }
.chip.bad { background: rgba(245,246,247,0.08); color: #F5F6F7; }

.footer { color: #8A8F98; font-size: 0.8rem; margin-top: 2.5rem;
          padding-top: 1rem; border-top: 1px solid #26262B; }

button[data-baseweb="tab"] { font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def card(label, value, sub="", gold=False):
    cls = "value gold" if gold else "value"
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="{cls}">{value}</div><div class="sub">{sub}</div></div>')


def callout(text, muted=False):
    cls = "callout muted" if muted else "callout"
    return f'<div class="{cls}">{text}</div>'


def section(title, sub=""):
    st.markdown(f'<div class="section-title">{title}</div>'
                f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def style(fig, height=380, title=None):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", y=-0.18, font=dict(color=GREY)),
        hoverlabel=dict(bgcolor=PANEL, font_color=TEXT),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0,
                                     font=dict(size=15, color=TEXT)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False, color=GREY)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, color=GREY)
    return fig


@st.cache_data(ttl=3600, show_spinner="Loading 2 years of prices…")
def get_prices(tickers: tuple) -> pd.DataFrame:
    return load_prices(list(tickers))


def analyze(w: pd.Series):
    """Run the full AUREN engine on one portfolio. Returns None if data is too short."""
    prices = get_prices(tuple(sorted(w.index))).dropna()
    if len(prices) < 30:
        return None
    c = concentration(w)
    pr = portfolio_returns(prices, w)
    vol, mdd = volatility(pr), max_drawdown(pr)
    value = (1 + pr).cumprod()
    drawdown = value / value.cummax() - 1
    corr = correlation(prices, w.index)
    pairs = high_pairs(corr)
    exposure = w.groupby(lambda t: SECTORS[t]).sum().sort_values(ascending=False)
    score, level, hits = risk_score(c, vol, mdd, pairs, float(exposure.max()))
    return dict(prices=prices, c=c, pr=pr, vol=vol, mdd=mdd, value=value,
                drawdown=drawdown, corr=corr, pairs=pairs, exposure=exposure,
                score=score, level=level, hits=hits)


PF_COLUMNS = {
    "ticker": st.column_config.SelectboxColumn(
        "Ticker", options=sorted(SECTORS), required=True),
    "weight": st.column_config.NumberColumn(
        "Weight %", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
}

# ---------------------------------------------------------- session state
DEFAULT_PF = pd.DataFrame({"ticker": ["NVDA", "MSFT", "GOOGL", "ETH-USD"],
                           "weight": [40.0, 30.0, 20.0, 10.0]})
st.session_state.setdefault("pf_df", DEFAULT_PF)
st.session_state.setdefault("pf_v", 0)
st.session_state.setdefault("size", 10000.0)
st.session_state.setdefault("wallet_msg", None)
st.session_state.setdefault("wallet_info", None)


def load_wallet():
    """Button callback: read a public Base address and fill the portfolio."""
    addr = st.session_state.get("wallet_addr", "").strip()
    st.session_state.wallet_info = None
    if not is_valid_address(addr):
        st.session_state.wallet_msg = (
            "error", "Enter a valid Base address: 0x followed by 40 characters.")
        return
    try:
        info = wallet_holdings(addr, set(SECTORS),
                               setting("BASE_RPC_URL", BASE_RPC))  # CHANGED (Groq)
    except Exception as e:
        st.session_state.wallet_msg = (
            "error", f"Could not read the wallet right now ({type(e).__name__}). "
                     "Try again in a minute.")
        return
    st.session_state.wallet_info = info
    df, total, skipped = holdings_to_portfolio(info["rows"])
    if df.empty:
        st.session_state.wallet_msg = (
            "warning", "No supported tokenized stocks found in this wallet.")
        return
    st.session_state.pf_df = df.reset_index(drop=True)
    st.session_state.pf_v += 1
    st.session_state.size = max(100.0, round(total, 2))
    extra = f" {len(skipped)} other holding(s) skipped." if skipped else ""
    st.session_state.wallet_msg = (
        "success", f"Loaded {len(df)} supported holding(s), about ${total:,.0f}.{extra}")


# ------------------------------------------------------------------- hero
st.markdown('<div class="wordmark"><span class="ring">◐</span>AUREN</div>'
            '<div class="tagline">See the risk before it sees you.</div>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="wordmark small"><span class="ring">◐</span>AUREN</div>',
                unsafe_allow_html=True)
    st.markdown("##### Your portfolio")

    pf = st.data_editor(
        st.session_state.pf_df, num_rows="dynamic", hide_index=True,
        key=f"pf_{st.session_state.pf_v}", column_config=PF_COLUMNS)
    pf = pf.dropna(subset=["ticker", "weight"])
    total = float(pf["weight"].sum())
    ok = abs(total - 100) <= 0.01
    st.markdown(f'<span class="chip {"ok" if ok else "bad"}">Total: {total:.1f}%</span>',
                unsafe_allow_html=True)
    st.number_input("Portfolio size ($)", min_value=100.0, step=500.0, key="size")
    st.caption("Weights must add up to 100%.")

    with st.expander("Base wallet (read-only)"):
        st.text_input("Public address", key="wallet_addr", placeholder="0x…")
        st.button("Load tokenized stocks", on_click=load_wallet)
        msg = st.session_state.wallet_msg
        if msg:
            {"error": st.error, "warning": st.warning,
             "success": st.success}[msg[0]](msg[1])
        st.caption("Read-only: AUREN only reads public balances. "
                   "Never paste a private key or seed phrase. "
                   "Addresses are not stored.")
    st.caption("Analytics only, not investment advice.")

# ------------------------------------------------------------- validation
if pf.empty:
    st.info("Add at least one holding in the sidebar to begin.")
    st.stop()
if pf["ticker"].duplicated().any():
    st.error("Each ticker can appear only once. Merge the duplicate rows in the sidebar.")
    st.stop()
if not ok:
    st.error(f"Weights add up to {total:.1f}%. They need to add up to 100%.")
    st.stop()

weights = pd.Series(pf["weight"].values / 100, index=pf["ticker"].values)
n = len(weights)
size = float(st.session_state.size)

A = analyze(weights)
if A is None:
    st.error("Not enough price history for these tickers. Try other holdings.")
    st.stop()
prices, c, pr, vol, mdd = A["prices"], A["c"], A["pr"], A["vol"], A["mdd"]
value, drawdown, corr, pairs = A["value"], A["drawdown"], A["corr"], A["pairs"]
exposure, score, level, hits = A["exposure"], A["score"], A["level"], A["hits"]

# live score card in the sidebar
with st.sidebar:
    st.markdown(
        '<div class="card" style="text-align:center;margin-top:14px">'
        '<div class="label">Risk score</div>'
        f'<div class="value gold" style="font-size:2.4rem">{score}'
        '<span style="font-size:0.9rem;color:#8A8F98"> / 100</span></div>'
        f'<div class="sub">{level}</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------- tabs
# NEW: added the "Concentration" tab
(tab_pf, tab_conc, tab_score, tab_risk,
 tab_corr, tab_whatif, tab_ai) = st.tabs(
    ["Portfolio", "Concentration", "Risk Overview", "Risk",
     "Correlation", "What-if", "AI Report"])

# ---- Portfolio (MVP 1-3)
with tab_pf:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(card("Portfolio value", f"${size:,.0f}", "Size set in the sidebar"),
                unsafe_allow_html=True)
    k2.markdown(card("Largest holding",
                     f"{c['top_asset']} · {c['top_weight']:.0%}",
                     "How much rides on one holding"), unsafe_allow_html=True)
    k3.markdown(card("Effective assets", f"{c['effective_assets']:.1f}",
                     f"Behaves like {c['effective_assets']:.1f} equal holdings, "
                     f"not {n}", gold=True), unsafe_allow_html=True)
    k4.markdown(card("Top sector", f"{exposure.max():.0%}", exposure.idxmax()),
                unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])
    with left:
        section("Holdings", "Each asset, its sector and its dollar value")
        holdings = pd.DataFrame({
            "Ticker": weights.index,
            "Sector": [SECTORS[t] for t in weights.index],
            "Weight %": weights.values * 100,
            "Value ($)": weights.values * size,
        })
        st.dataframe(
            holdings, hide_index=True,
            column_config={
                "Weight %": st.column_config.ProgressColumn(
                    "Weight %", min_value=0, max_value=100, format="%.1f"),
                "Value ($)": st.column_config.NumberColumn("Value ($)", format="$%.0f"),
            })
    with right:
        section("Allocation", "Share of the portfolio by holding")
        donut_colors = [GOLD if t == c["top_asset"] else ASSET_GREYS[i % len(ASSET_GREYS)]
                        for i, t in enumerate(weights.index)]
        fig = go.Figure(go.Pie(
            labels=list(weights.index), values=list(weights.values), hole=0.62,
            sort=False, marker=dict(colors=donut_colors, line=dict(color=BG, width=2)),
            textinfo="label+percent", textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{label}: %{percent}<extra></extra>"))
        fig.add_annotation(text=f"${size:,.0f}", showarrow=False,
                           font=dict(size=20, color=TEXT, family="JetBrains Mono"))
        style(fig, height=320)
        fig.update_layout(showlegend=False)
        with st.container(border=True):
            st.plotly_chart(fig)

    section("Sector exposure", "Where the portfolio is really invested")
    ex = exposure.sort_values()
    colors = [GOLD if s == exposure.idxmax() else "#5B5F68" for s in ex.index]
    fig = go.Figure(go.Bar(
        x=ex.values, y=ex.index, orientation="h", marker_color=colors,
        text=[f"{v:.0%}" for v in ex.values], textposition="outside",
        textfont=dict(color=TEXT), hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    style(fig, height=max(220, 60 * len(ex) + 60))
    fig.update_xaxes(tickformat=".0%", range=[0, min(1.0, ex.max() * 1.2)])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    with st.container(border=True):
        st.plotly_chart(fig)

    info = st.session_state.wallet_info
    if info and info["rows"]:
        with st.expander("Tokenized stocks detected in the wallet"):
            det = pd.DataFrame([{
                "Token": r["token"], "Underlying": r["ticker"],
                "Tokens": r["tokens"], "Shares": r["shares"],
                "Price ($)": r["price"], "Value ($)": r["value"],
                "Analysed": "Yes" if r["supported"] and r["value"] > 0 else "No",
            } for r in info["rows"]])
            st.dataframe(det, hide_index=True)
            st.caption("Value = token balance × Chainlink reference price. Shares apply "
                       "the token's multiplier. Reference prices can be stale outside "
                       "market hours. Only holdings marked Yes are analysed.")
    elif info is not None:
        st.caption(f"Wallet read worked ({info['n_tokens']} tokenized stocks checked), "
                   "but none are held at this address.")

# ---- NEW: Concentration
with tab_conc:
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.markdown(card("Largest holding",
                      f"{c['top_asset']} · {c['top_weight']:.0%}",
                      "Over 30% adds risk points"), unsafe_allow_html=True)
    cc2.markdown(card("Top 3 holdings", f"{c['top3_weight']:.0%}",
                      "Share held in the three biggest positions"),
                 unsafe_allow_html=True)
    cc3.markdown(card("Effective assets", f"{c['effective_assets']:.1f}",
                      f"Behaves like {c['effective_assets']:.1f} equal holdings, "
                      f"not {n}", gold=True), unsafe_allow_html=True)
    cc4.markdown(card("Concentration index", f"{c['hhi']:.2f}",
                      "HHI. 1.00 means everything in one asset"),
                 unsafe_allow_html=True)

    section("Holdings by weight",
            "The dashed line is AUREN's 30% single-holding rule")
    wts = weights.sort_values()
    bar_colors = [GOLD if v > 0.30 else "#5B5F68" for v in wts.values]
    fig = go.Figure(go.Bar(
        x=wts.values, y=wts.index, orientation="h", marker_color=bar_colors,
        text=[f"{v:.0%}" for v in wts.values], textposition="outside",
        textfont=dict(color=TEXT), hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.add_vline(x=0.30, line_dash="dash", line_color=GREY,
                  annotation_text="30% rule", annotation_font_color=GREY)
    style(fig, height=max(240, 60 * len(wts) + 60))
    fig.update_xaxes(tickformat=".0%",
                     range=[0, max(0.35, min(1.0, wts.max() * 1.25))])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    with st.container(border=True):
        st.plotly_chart(fig)

    if c["top_weight"] > 0.30:
        st.markdown(callout(
            f"<b>{c['top_asset']}</b> is {c['top_weight']:.0%} of the portfolio, "
            f"above the 30% line. A lot rides on one holding."),
            unsafe_allow_html=True)
    else:
        st.markdown(callout(
            f"No single holding is above 30%. The largest, <b>{c['top_asset']}</b>, "
            f"is {c['top_weight']:.0%}.", muted=True), unsafe_allow_html=True)
    drop = 0.20
    st.markdown(callout(
        f"<b>Illustration:</b> if {c['top_asset']} fell {drop:.0%}, the portfolio would "
        f"lose about {c['top_weight'] * drop:.1%} from that one holding alone. "
        f"This is simple arithmetic, not a prediction.", muted=True),
        unsafe_allow_html=True)

    with st.expander("Show the maths"):
        terms = " + ".join(f"{w:.2f}²" for w in weights.sort_values(ascending=False))
        st.markdown(
            f"**Your numbers:** {terms} = **{c['hhi']:.2f}** (the concentration index)\n\n"
            f"**Effective assets** = 1 ÷ {c['hhi']:.2f} = **{c['effective_assets']:.1f}**")
        st.caption("Square each weight and add them up to get the index (HHI). "
                   "Dividing 1 by it tells you how many equal-sized holdings the "
                   "portfolio behaves like. Equal weights give exactly the number of assets.")
    st.caption("Concentration uses today's weights, not past prices. "
               "Analytics only, not investment advice.")

# ---- Risk Overview
with tab_score:
    left, right = st.columns([1, 1.6])
    with left:
        st.markdown(
            '<div class="card" style="text-align:center">'
            '<div class="label">Risk score</div>'
            f'<div class="value gold" style="font-size:4.4rem">{score}'
            '<span style="font-size:1.3rem;color:#8A8F98"> / 100</span></div>'
            f'<div class="sub">Level: <b style="color:#F5F6F7">{level}</b></div>'
            '</div>', unsafe_allow_html=True)
        st.progress(score / 100)
    with right:
        section("Why this score", "Every point comes with a reason")
        if hits:
            for pts, why in hits:
                st.markdown(callout(f"<b>+{pts}</b> · {why}"), unsafe_allow_html=True)
        else:
            st.markdown(callout("No rules triggered for this portfolio.", muted=True),
                        unsafe_allow_html=True)
    with st.expander("See every rule"):
        st.dataframe(pd.DataFrame(RULE_TABLE, columns=["Points", "Rule"]),
                     hide_index=True)
        st.caption("The thresholds are judgement calls, not laws of finance. "
                   "They are public so you can disagree with them.")
    st.caption("A rule-based measure of past data, not a prediction. "
               "Analytics only, not investment advice.")

# ---- Risk (MVP 5)
with tab_risk:
    r1, r2, r3 = st.columns(3)
    r1.markdown(card("Volatility (annualised)", f"{vol:.1%}",
                     "How much the value swings in a typical year", gold=True),
                unsafe_allow_html=True)
    r2.markdown(card("Worst drawdown", f"{mdd:.1%}",
                     f"Peak-to-bottom fall, trough on {drawdown.idxmin():%d %b %Y}"),
                unsafe_allow_html=True)
    r3.markdown(card("Days analysed", f"{len(pr)}",
                     "Trading days where every asset has a price"),
                unsafe_allow_html=True)

    section("Price history", "Every asset rebased to 100. Gold is your portfolio.")
    norm = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], name=col, mode="lines",
            line=dict(width=1.3, color=ASSET_GREYS[i % len(ASSET_GREYS)])))
    fig.add_trace(go.Scatter(
        x=value.index, y=value.values * 100, name="Portfolio", mode="lines",
        line=dict(width=3, color=GOLD)))
    style(fig, height=380)
    with st.container(border=True):
        st.plotly_chart(fig)

    section("Drawdown", "How far the portfolio sat below its previous peak")
    trough = drawdown.idxmin()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values, mode="lines", name="Drawdown",
        line=dict(color=GOLD, width=2), fill="tozeroy",
        fillcolor="rgba(227,179,65,0.15)",
        hovertemplate="%{x|%d %b %Y}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[trough], y=[mdd], mode="markers+text", showlegend=False,
        text=[f"{mdd:.1%}"], textposition="bottom center",
        textfont=dict(color=GOLD, size=13), marker=dict(color=GOLD, size=9)))
    style(fig, height=330)
    fig.update_yaxes(tickformat=".0%", range=[mdd * 1.3, 0.02])
    with st.container(border=True):
        st.plotly_chart(fig)
    st.caption("Based on past data. Past data does not predict the future. "
               "Analytics only, not investment advice.")

# ---- Correlation (MVP 4)
with tab_corr:
    cols = list(corr.columns)
    all_pairs = [(a, b, float(corr.loc[a, b]))
                 for i, a in enumerate(cols) for b in cols[i + 1:]]
    top = max(all_pairs, key=lambda p: p[2]) if all_pairs else None

    q1, q2 = st.columns(2)
    if top:
        q1.markdown(card("Most correlated pair", f"{top[0]} · {top[1]}",
                         f"Correlation {top[2]:.2f}"), unsafe_allow_html=True)
    else:
        q1.markdown(card("Most correlated pair", "–",
                         "Add a second holding to compare"), unsafe_allow_html=True)
    q2.markdown(card("Pairs above 0.7", f"{len(pairs)}",
                     f"Out of {len(all_pairs)} pairs", gold=True),
                unsafe_allow_html=True)

    section("Correlation heatmap",
            "Brighter gold means the assets tend to move together")
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale=["#18181B", GOLD])
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont=dict(size=14))
    style(fig, height=420)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    with st.container(border=True):
        st.plotly_chart(fig)

    if pairs:
        for a, b, v in pairs:
            st.markdown(callout(f"<b>{a}</b> and <b>{b}</b> move together ({v}). "
                                f"They may fall together."), unsafe_allow_html=True)
    else:
        st.markdown(callout("No pair is above 0.7 correlation in this portfolio.",
                            muted=True), unsafe_allow_html=True)
    st.caption("Based on past data. Past data does not predict the future. "
               "Analytics only, not investment advice.")

# ---- What-if (MVP 7)
with tab_whatif:
    section("What if the allocation changed?",
            "Edit the second portfolio below. AUREN compares the risk side by side.")
    sig = str(abs(hash(tuple(zip(weights.index, weights.round(4))))))
    base_b = pd.DataFrame({"ticker": list(weights.index),
                           "weight": list(weights.values * 100)})
    wb_df = st.data_editor(base_b, num_rows="dynamic", hide_index=True,
                           key=f"pfb_{sig}", column_config=PF_COLUMNS)
    wb_df = wb_df.dropna(subset=["ticker", "weight"])
    tot_b = float(wb_df["weight"].sum())
    ok_b = abs(tot_b - 100) <= 0.01
    st.markdown(f'<span class="chip {"ok" if ok_b else "bad"}">What-if total: '
                f'{tot_b:.1f}%</span>', unsafe_allow_html=True)

    if wb_df.empty or wb_df["ticker"].duplicated().any():
        st.info("Add holdings with no duplicate tickers to compare.")
    elif not ok_b:
        st.warning("What-if weights must add up to 100%.")
    else:
        wb = pd.Series(wb_df["weight"].values / 100, index=wb_df["ticker"].values)
        B = analyze(wb)
        if B is None:
            st.error("Not enough price history for the what-if tickers.")
        else:
            d = B["score"] - A["score"]
            note = ("Lower risk score" if d < 0 else
                    "Higher risk score" if d > 0 else "No change")
            w1, w2, w3 = st.columns(3)
            w1.markdown(card("Current score", f"{A['score']}", A["level"]),
                        unsafe_allow_html=True)
            w2.markdown(card("What-if score", f"{B['score']}", B["level"], gold=True),
                        unsafe_allow_html=True)
            w3.markdown(card("Change", f"{d:+d}", note), unsafe_allow_html=True)

            section("Side by side")
            rows = [
                ("Risk score", f"{A['score']} ({A['level']})", f"{B['score']} ({B['level']})"),
                ("Largest holding", f"{A['c']['top_asset']} · {A['c']['top_weight']:.0%}",
                 f"{B['c']['top_asset']} · {B['c']['top_weight']:.0%}"),
                # NEW: concentration rows
                ("Top 3 holdings", f"{A['c']['top3_weight']:.0%}",
                 f"{B['c']['top3_weight']:.0%}"),
                ("Effective number of assets", f"{A['c']['effective_assets']:.1f}",
                 f"{B['c']['effective_assets']:.1f}"),
                ("Top sector", f"{A['exposure'].idxmax()} · {A['exposure'].max():.0%}",
                 f"{B['exposure'].idxmax()} · {B['exposure'].max():.0%}"),
                ("Volatility (annualised)", f"{A['vol']:.1%}", f"{B['vol']:.1%}"),
                ("Worst drawdown", f"{A['mdd']:.1%}", f"{B['mdd']:.1%}"),
                ("Pairs above 0.7", str(len(A["pairs"])), str(len(B["pairs"]))),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Metric", "Current", "What-if"]),
                         hide_index=True)

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Current: reasons**")
                for pts, why in A["hits"]:
                    st.markdown(callout(f"<b>+{pts}</b> · {why}", muted=True),
                                unsafe_allow_html=True)
                if not A["hits"]:
                    st.caption("No rules triggered.")
            with rc2:
                st.markdown("**What-if: reasons**")
                for pts, why in B["hits"]:
                    st.markdown(callout(f"<b>+{pts}</b> · {why}"),
                                unsafe_allow_html=True)
                if not B["hits"]:
                    st.caption("No rules triggered.")

            section("Past drawdown: current vs what-if",
                    "Weights held constant. Based on the last 2 years of data.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=A["drawdown"].index, y=A["drawdown"].values,
                                     name="Current", line=dict(color="#7A7F88", width=2)))
            fig.add_trace(go.Scatter(x=B["drawdown"].index, y=B["drawdown"].values,
                                     name="What-if", line=dict(color=GOLD, width=2.5)))
            style(fig, height=320)
            fig.update_yaxes(tickformat=".0%")
            with st.container(border=True):
                st.plotly_chart(fig)
    st.caption("A comparison of past-data measures, not a prediction. "
               "Analytics only, not investment advice.")

# ---- AI Report (MVP 6)
with tab_ai:
    section("AI risk report",
            "AI explains. Maths measures. Every number comes from the tested engine.")
    metrics = {
        "risk_score": score, "risk_level": level,
        "score_reasons": [why for _, why in hits],
        "portfolio_value_usd": round(size, 2),
        "holdings_weight": {t: round(float(w), 4) for t, w in weights.items()},
        "largest_holding": {c["top_asset"]: round(c["top_weight"], 4)},
        "top_3_holdings_weight": round(c["top3_weight"], 4),          # NEW
        "concentration_index_hhi": round(c["hhi"], 4),                # NEW
        "effective_assets": round(c["effective_assets"], 2),
        "volatility_annualised": round(vol, 4),
        "max_drawdown": round(mdd, 4),
        "sector_exposure": {k: round(float(v), 4) for k, v in exposure.items()},
        "correlated_pairs_above_0.7": [[a, b, v] for a, b, v in pairs],
    }
    metrics_json = json.dumps(metrics, sort_keys=True, default=str)

    # CHANGED (Groq): config comes from .env or Streamlit secrets
    base_url = setting("LLM_BASE_URL")
    model = setting("LLM_MODEL")
    api_key = setting("LLM_API_KEY", "")
    configured = bool(base_url and model)

    if not configured:
        st.info("Set LLM_BASE_URL and LLM_MODEL (and LLM_API_KEY for hosted "
                "providers) in your .env file to turn on AI reports.")
    if st.button("Generate report", disabled=not configured):
        try:
            with st.spinner("Writing the report…"):
                text = ai_report(metrics, base_url, api_key, model)
            st.session_state.report = (metrics_json, text)
        except Exception as e:
            st.error(f"Could not generate the report ({type(e).__name__}). "
                     "Check LLM_BASE_URL, LLM_API_KEY and LLM_MODEL. "
                     "If you use Ollama, make sure it is running.")

    rep = st.session_state.get("report")
    if rep:
        if rep[0] != metrics_json:
            st.markdown(callout("The portfolio changed since this report was written. "
                                "Generate it again.", muted=True), unsafe_allow_html=True)
        st.markdown(rep[1])
        bad = unverified_numbers(rep[1], metrics)
        if bad:
            st.warning("These numbers in the report were not found in the data: "
                       + ", ".join(f"{b:g}" for b in bad)
                       + ". Treat them with caution.")
        else:
            st.caption("No numbers outside the data were found. "
                       "Small whole numbers are not checked.")
        reasons = [f"- +{p} · {why}" for p, why in hits] or ["- No rules triggered"]
        md = "\n".join([
            "# AUREN risk report", "",
            f"**Risk score:** {score} / 100 ({level})", "",
            "## Why this score", *reasons, "",
            "## Key numbers",
            f"- Largest holding: {c['top_asset']} ({c['top_weight']:.0%})",
            f"- Top 3 holdings: {c['top3_weight']:.0%}",
            f"- Effective number of assets: {c['effective_assets']:.1f}",
            f"- Volatility (annualised): {vol:.1%}",
            f"- Worst drawdown: {mdd:.1%}", "",
            "## AI explanation", rep[1], "",
            "_Analytics only, not investment advice._"])
        st.download_button("Download report", md, "auren-report.md")
    with st.expander("Numbers sent to the AI (check them)"):
        st.json(metrics)

# ----------------------------------------------------------------- footer
st.markdown('<div class="footer">Analytics only, not investment advice. '
            'Past data does not predict the future. Price data from Yahoo '
            'Finance via yfinance (unofficial, for demo analytics only). '
            'Tokenized stock data from the public Base and Coinbase APIs.</div>',
            unsafe_allow_html=True)