import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

st.set_page_config(page_title="VyaparIQ", page_icon="🌾", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 0 !important; max-width: 100% !important;}

/* Navbar */
.mm-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 32px;
    border-bottom: 0.5px solid #e5e5e5;
    background: #fff;
    position: sticky;
    top: 0;
    z-index: 100;
}
.mm-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 600;
    color: #111;
}
.mm-logo-icon {
    width: 32px;
    height: 32px;
    background: #1a6b3c;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 16px;
    font-weight: 700;
}
.mm-badge {
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 20px;
    border: 0.5px solid #e5e5e5;
    color: #666;
    background: #fafafa;
}

/* Ticker */
.ticker-outer {
    overflow: hidden;
    flex: 1;
    margin: 0 32px;
}
.ticker-track {
    display: flex;
    gap: 40px;
    white-space: nowrap;
    animation: ticker 22s linear infinite;
}
.ticker-item {
    font-size: 13px;
    color: #555;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.ticker-price { font-weight: 600; color: #111; }
.ticker-up   { color: #1a6b3c; font-size: 12px; }
.ticker-down { color: #c0392b; font-size: 12px; }
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* Hero */
.mm-hero {
    padding: 36px 32px 24px;
    border-bottom: 0.5px solid #e5e5e5;
    background: #fff;
}
.mm-hero-label {
    font-size: 11px;
    color: #999;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.mm-hero-title {
    font-size: 26px;
    font-weight: 600;
    color: #111;
    margin-bottom: 4px;
}
.mm-hero-sub {
    font-size: 15px;
    color: #666;
}

/* Cards */
.mm-card {
    background: #fff;
    border: 0.5px solid #e5e5e5;
    border-radius: 12px;
    padding: 20px 24px;
    height: 100%;
}
.mm-card-layer {
    font-size: 11px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.mm-card-title {
    font-size: 15px;
    font-weight: 600;
    color: #111;
    margin-bottom: 18px;
}
.mm-price-big {
    font-size: 38px;
    font-weight: 600;
    color: #111;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
}
.mm-price-range {
    font-size: 13px;
    color: #888;
    margin-bottom: 14px;
}
.mm-badge-green {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 20px;
    background: #eaf3de;
    color: #3b6d11;
    border: 0.5px solid #c0dd97;
}
.mm-badge-warn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 20px;
    background: #faeeda;
    color: #854f0b;
    border: 0.5px solid #fac775;
}
.mm-stat-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 0.5px solid #f0f0f0;
    font-size: 13px;
}
.mm-stat-row:last-child { border-bottom: none; }
.mm-stat-key { color: #888; }
.mm-stat-val { font-weight: 500; color: #111; }

/* Signal */
.signal-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 16px;
    border-radius: 10px;
    margin-bottom: 14px;
}
.signal-box.buy  { background: #eaf3de; }
.signal-box.wait { background: #f5f5f5; }
.signal-box.pause{ background: #faeeda; }
.signal-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-green  { background: #1a6b3c; }
.dot-blue   { background: #378add; }
.dot-amber  { background: #ef9f27; }
.signal-label {
    font-size: 14px;
    font-weight: 600;
}
.signal-meta {
    font-size: 13px;
    color: #555;
    line-height: 1.7;
}

/* Footer */
.mm-footer {
    padding: 14px 32px;
    border-top: 0.5px solid #e5e5e5;
    font-size: 12px;
    color: #aaa;
    display: flex;
    justify-content: space-between;
    background: #fff;
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data_clean.csv', parse_dates=['date'])

@st.cache_resource
def load_model():
    model    = joblib.load('model_lr.pkl')
    encoders = joblib.load('label_encoders.pkl')
    features = joblib.load('features.pkl')
    return model, encoders, features

df = load_data()
model, encoders, features = load_model()

# ── Build ticker from real data ───────────────────────────────────────────
@st.cache_data
def build_ticker(df):
    # Get one latest row per district+variety combination
    # This ensures all districts are represented
    idx = (df.groupby(['district', 'variety'], observed=True)['date']
             .idxmax())
    latest_per_group = df.loc[idx].copy()
    
    # Get previous price for each group
    items = []
    for _, row in latest_per_group.iterrows():
        group_data = df[
            (df['district'] == row['district']) &
            (df['variety']  == row['variety'])
        ].sort_values('date')
        
        if len(group_data) >= 2:
            prev_price = group_data.iloc[-2]['modal_price']
            change     = row['modal_price'] - prev_price
        else:
            change = 0
            
        sign = '+' if change >= 0 else ''
        cls  = 'ticker-up' if change >= 0 else 'ticker-down'
        items.append(
            f'<span class="ticker-item">{row["district"]} {row["variety"]} '
            f'<span class="ticker-price">₹{row["modal_price"]:,.0f}</span>'
            f'<span class="{cls}">{sign}{change:.0f}</span></span>'
        )
    
    # Shuffle for variety then duplicate for infinite scroll
    import random
    random.shuffle(items)
    return ''.join(items * 2)

ticker_html = build_ticker(df)

# ── Navbar ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mm-nav">
  <div class="mm-logo">
    <div class="mm-logo-icon">M</div>
    VyaparIQ
  </div>
  <div class="ticker-outer">
    <div class="ticker-track">{ticker_html}</div>
  </div>
  <div class="mm-badge">MP Mandis — Nov to Yesterday</div>
</div>
""", unsafe_allow_html=True)

# ── Hero + Selectors ──────────────────────────────────────────────────────
st.markdown("""
<div class="mm-hero">
  <div class="mm-hero-label">Wheat Price Intelligence</div>
  <div class="mm-hero-title">Select your market</div>
  <div class="mm-hero-sub">Tomorrow's price prediction and buying signal for MP mandis</div>
</div>
""", unsafe_allow_html=True)

col_d, col_v, col_g, col_empty = st.columns([1, 1, 1, 3])
districts = ["-- Select --"] + sorted(df['district'].astype(str).unique())
varieties = ["-- Select --"] + sorted(df['variety'].astype(str).unique())
grades    = ["-- Select --"] + sorted(df['grade'].astype(str).unique())

with col_d: selected_district = st.selectbox("District", districts)
with col_v: selected_variety  = st.selectbox("Variety",  varieties)
with col_g: selected_grade    = st.selectbox("Grade",    grades)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if "-- Select --" in [selected_district, selected_variety, selected_grade]:
    st.markdown("""
    <div style='padding:48px 32px;text-align:center;color:#aaa;font-size:15px'>
        Select district, variety, and grade above to see predictions
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Filter data ───────────────────────────────────────────────────────────
market_df = df[
    (df['district'].astype(str) == selected_district) &
    (df['variety'].astype(str)  == selected_variety)  &
    (df['grade'].astype(str)    == selected_grade)
].sort_values('date')

if len(market_df) == 0:
    st.warning(f"No data for {selected_district} — {selected_variety} — {selected_grade}")
    st.stop()

latest      = market_df.iloc[-1]
latest_date = latest['date']

# ── Encode and predict ────────────────────────────────────────────────────
def encode_row(row):
    r = row.copy()
    for col in ['district', 'variety', 'grade']:
        le  = encoders[col]
        val = str(row[col])
        r[col] = le.transform([val])[0] if val in le.classes_ else 0
    return r

try:
    X_pred          = pd.DataFrame([encode_row(latest)[features]])
    X_pred          = X_pred.fillna(X_pred.median())
    predicted_price = model.predict(X_pred)[0]
    MAE             = 86.28
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# ── Layer 2 signal ────────────────────────────────────────────────────────
THRESHOLD   = 60
MSP         = 2425
GOOD_MONTHS = [1, 3, 5, 11, 12]

price_lag1       = latest['price_lag1']
price_roll3_mean = latest['price_roll3_mean']
current_month    = latest['date'].month
gap = (price_lag1 - price_roll3_mean
       if pd.notna(price_lag1) and pd.notna(price_roll3_mean) else 0)

buy_signal    = gap < -THRESHOLD and price_lag1 > MSP and current_month in GOOD_MONTHS
signal_paused = current_month not in GOOD_MONTHS

# ── Cards ─────────────────────────────────────────────────────────────────
st.markdown("<div style='padding:24px 32px 0'>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    above_msp = predicted_price - MSP
    badge = (f'<span class="mm-badge-green">✓ ₹{above_msp:.0f} above MSP</span>'
             if above_msp >= 0
             else f'<span class="mm-badge-warn">⚠ ₹{abs(above_msp):.0f} below MSP</span>')

    st.markdown(f"""
    <div class="mm-card">
      <div class="mm-card-layer">Layer 1</div>
      <div class="mm-card-title">Price prediction</div>
      <div class="mm-price-big">₹{predicted_price:,.0f}</div>
      <div class="mm-price-range">Range ₹{predicted_price-MAE:,.0f} — ₹{predicted_price+MAE:,.0f} per quintal</div>
      {badge}
      <div style="margin-top:18px">
        <div class="mm-stat-row">
          <span class="mm-stat-key">Data up to</span>
          <span class="mm-stat-val">{latest_date.strftime('%d %b %Y')}</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Model</span>
          <span class="mm-stat-val">Linear Regression</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Avg error (MAE)</span>
          <span class="mm-stat-val">±₹86 / quintal</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">MSP floor</span>
          <span class="mm-stat-val">₹{MSP:,}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    if buy_signal:
        box_cls, dot_cls, label, color = "buy", "dot-green", "Buy signal — consider buying today", "#3b6d11"
        meta = f"""Yesterday's price ₹{price_lag1:,.0f}<br>
        3-day average ₹{price_roll3_mean:,.0f}<br>
        Depression ₹{abs(gap):.0f} below average<br>
        Historical recovery rate <strong>67.3%</strong> within 3 days"""
    elif signal_paused:
        box_cls, dot_cls, label, color = "pause", "dot-amber", "Signal paused this month", "#854f0b"
        meta = "Feb and Apr have below 50% recovery rate.<br>VyaparIQ suppresses signals in these months."
    else:
        box_cls, dot_cls, label, color = "wait", "dot-blue", "Wait — no opportunity today", "#378add"
        meta = f"""Yesterday's price ₹{price_lag1:,.0f}<br>
        3-day average ₹{price_roll3_mean:,.0f}<br>
        Gap ₹{abs(gap):.0f} — need ₹{THRESHOLD}+ to trigger signal"""

    st.markdown(f"""
    <div class="mm-card">
      <div class="mm-card-layer">Layer 2</div>
      <div class="mm-card-title">Buying signal</div>
      <div class="signal-box {box_cls}">
        <div class="signal-dot {dot_cls}"></div>
        <div class="signal-label" style="color:{color}">{label}</div>
      </div>
      <div class="signal-meta">{meta}</div>
      <div style="margin-top:18px">
        <div class="mm-stat-row">
          <span class="mm-stat-key">Active months</span>
          <span class="mm-stat-val">Nov, Dec, Jan, Mar, May</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Recovery rate (T+3)</span>
          <span class="mm-stat-val">67.3%</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Avg gain on signal</span>
          <span class="mm-stat-val">₹34 / quintal</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Depression threshold</span>
          <span class="mm-stat-val">₹{THRESHOLD} below 3-day avg</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Price Trend Chart ─────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:24px 32px 8px'>
  <div style='font-size:15px;font-weight:600;color:#111;margin-bottom:4px'>
    Price trend — {selected_district}, {selected_variety} ({selected_grade})
  </div>
  <div style='font-size:13px;color:#888;margin-bottom:16px'>Last 60 trading days</div>
""", unsafe_allow_html=True)

recent  = market_df.tail(60)
sig_pts = recent[
    (recent['price_lag1'] - recent['price_roll3_mean'] < -THRESHOLD) &
    (recent['price_lag1'] > MSP) &
    (recent['date'].dt.month.isin(GOOD_MONTHS))
]

fig, ax = plt.subplots(figsize=(13, 3.5))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

ax.plot(recent['date'], recent['modal_price'],
        color='#378add', linewidth=1.8, label='Modal price')
ax.plot(recent['date'], recent['price_roll3_mean'],
        color='#ef9f27', linewidth=1.4, linestyle='--', label='3-day avg')
ax.axhline(MSP, color='#ccc', linewidth=0.8, linestyle=':')
ax.text(recent['date'].iloc[0], MSP + 8, 'MSP ₹2425',
        fontsize=9, color='#aaa')

if len(sig_pts):
    ax.scatter(sig_pts['date'], sig_pts['price_lag1'],
               color='#1a6b3c', s=60, zorder=5,
               marker='^', label='Buy signal')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#eee')
ax.spines['bottom'].set_color('#eee')
ax.tick_params(colors='#aaa', labelsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
ax.set_ylabel('')
ax.legend(loc='upper right', fontsize=10,
          framealpha=0, labelcolor='#555')
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mm-footer">
  <span>VyaparIQ v1.0 — Regular wheat · MP mandis · Nov–May season only</span>
  <span>Data: agmarknet.gov.in</span>
</div>
""", unsafe_allow_html=True)
