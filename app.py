import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

st.set_page_config(page_title="VyaparIQ", page_icon="🌾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 0 !important; max-width: 100% !important;}

* { font-family: 'Outfit', sans-serif !important; }

.mm-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 40px;
    border-bottom: 3px solid #D97706;
    background: #1B4332;
    position: sticky;
    top: 0;
    z-index: 100;
}
.mm-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}
.mm-logo-icon {
    width: 38px;
    height: 38px;
    background: #D97706;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 20px;
}
.mm-badge {
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 30px;
    border: 1px solid rgba(255,255,255,0.2);
    color: #F3F4F6;
    background: rgba(255,255,255,0.1);
}
.ticker-outer {
    overflow: hidden;
    flex: 1;
    margin: 0 40px;
    background: rgba(0,0,0,0.2);
    padding: 8px 16px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.05);
}
.ticker-track {
    display: flex;
    gap: 48px;
    white-space: nowrap;
    animation: ticker 22s linear infinite;
}
.ticker-item {
    font-size: 14px;
    color: #E5E7EB;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
}
.ticker-price { font-weight: 700; color: #FBBF24; }
.ticker-up   { color: #34D399; font-weight: 700; }
.ticker-down { color: #F87171; font-weight: 700; }
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.mm-hero {
    padding: 54px 40px 72px 40px;
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
    color: #FFFFFF;
    border-bottom: 1px solid #E5E7EB;
}
.mm-hero-label {
    font-size: 12px;
    color: #FBBF24;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.mm-hero-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 8px;
}
.mm-hero-sub {
    font-size: 16px;
    color: #D8F3DC;
    margin-bottom: 4px;
}
.mm-hero-date {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    margin-top: 6px;
}
.selector-container {
    background: #FFFFFF;
    margin: -28px 40px 32px 40px;
    padding: 24px 32px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    border: 1px solid #E5E7EB;
}
.mm-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    height: 100%;
}
.mm-card-layer {
    font-size: 12px;
    color: #D97706;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.mm-card-title {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 20px;
}
.mm-price-big {
    font-size: 46px;
    font-weight: 700;
    color: #1B4332;
    margin-bottom: 6px;
    letter-spacing: -1px;
}
.mm-price-range {
    font-size: 14px;
    color: #6B7280;
    margin-bottom: 18px;
    font-weight: 500;
}
.mm-badge-green {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 30px;
    background: #D8F3DC;
    color: #1B4332;
    border: 1px solid #B7E4C7;
}
.mm-badge-warn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 30px;
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}
.mm-stat-row {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid #F3F4F6;
    font-size: 14px;
}
.mm-stat-row:last-child { border-bottom: none; }
.mm-stat-key { color: #4B5563; font-weight: 500; }
.mm-stat-val { font-weight: 600; color: #111827; }
.signal-box {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
}
.signal-box.buy   { background: #D8F3DC; border-left: 5px solid #1B4332; }
.signal-box.wait  { background: #F3F4F6; border-left: 5px solid #9CA3AF; }
.signal-box.pause { background: #FEF3C7; border-left: 5px solid #D97706; }
.signal-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-green { background: #15803D; }
.dot-gray  { background: #6B7280; }
.dot-amber { background: #D97706; }
.signal-label {
    font-size: 15px;
    font-weight: 700;
}
.signal-meta {
    font-size: 14px;
    color: #374151;
    line-height: 1.8;
    background: #F9FAFB;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
}
.mm-footer {
    padding: 20px 40px;
    border-top: 1px solid #E5E7EB;
    font-size: 13px;
    font-weight: 500;
    color: #6B7280;
    display: flex;
    justify-content: space-between;
    background: #FFFFFF;
    margin-top: 48px;
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

# ── Build ticker ──────────────────────────────────────────────────────────
@st.cache_data
def build_ticker(df):
    idx = df.groupby(['district','variety'], observed=True)['date'].idxmax()
    latest_per_group = df.loc[idx].copy()
    items = []
    for _, row in latest_per_group.iterrows():
        group_data = df[
            (df['district'] == row['district']) &
            (df['variety']  == row['variety'])
        ].sort_values('date')
        change = row['modal_price'] - group_data.iloc[-2]['modal_price'] if len(group_data) >= 2 else 0
        sign   = '▲' if change >= 0 else '▼'
        cls    = 'ticker-up' if change >= 0 else 'ticker-down'
        items.append(
            f'<span class="ticker-item">{row["district"]} ({row["variety"]}) '
            f'<span class="ticker-price">₹{row["modal_price"]:,.0f}</span> '
            f'<span class="{cls}">{sign} {abs(change):.0f}</span></span>'
        )
    import random
    random.shuffle(items)
    return ''.join(items * 2)

ticker_html  = build_ticker(df)
latest_date  = df['date'].max()
last_updated = latest_date.strftime('%d %b %Y')

# ── Navbar ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mm-nav">
  <div class="mm-logo">
    <div class="mm-logo-icon">🌾</div>
    VyaparIQ
  </div>
  <div class="ticker-outer">
    <div class="ticker-track">{ticker_html}</div>
  </div>
  <div class="mm-badge">MP Mandi Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mm-hero">
  <div class="mm-hero-label">Krishi Bazaar Intelligence Platform</div>
  <div class="mm-hero-title">Select Your Mandi Market</div>
  <div class="mm-hero-sub">Real-time price predictions and buying signals for MP wheat mandis</div>
  <div class="mm-hero-date">Data updated up to {last_updated} · Updated daily from agmarknet.gov.in</div>
</div>
""", unsafe_allow_html=True)

# ── Selectors ─────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 16px 40px 8px 40px">', unsafe_allow_html=True)
col_d, col_v, col_g, col_empty = st.columns([1,1,1,2])

districts = ["-- Select --"] + sorted(df['district'].astype(str).unique())
varieties = ["-- Select --"] + sorted(df['variety'].astype(str).unique())
grades    = ["-- Select --"] + sorted(df['grade'].astype(str).unique())

with col_d: selected_district = st.selectbox("District", districts)
with col_v: selected_variety  = st.selectbox("Variety",  varieties)
with col_g: selected_grade    = st.selectbox("Grade",    grades)
st.markdown('</div>', unsafe_allow_html=True)

if "-- Select --" in [selected_district, selected_variety, selected_grade]:
    st.markdown("""
    <div style='padding:48px 40px;text-align:center;color:#9CA3AF;font-size:15px;font-weight:500'>
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
    st.warning(f"No data found for {selected_district} — {selected_variety} — {selected_grade}")
    st.stop()

latest = market_df.iloc[-1]

# ── Encode and predict ────────────────────────────────────────────────────
def encode_row(row):
    r = row.copy()
    for col in ['district','variety','grade']:
        le  = encoders[col]
        val = str(row[col])
        r[col] = le.transform([val])[0] if val in le.classes_ else 0
    return r

try:
    X_pred          = pd.DataFrame([encode_row(latest)[features]])
    X_pred          = X_pred.fillna(35.0)
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
st.markdown("<div style='padding:0 40px'>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    above_msp = predicted_price - MSP
    badge = (f'<span class="mm-badge-green">✓ ₹{above_msp:.0f} above MSP</span>'
             if above_msp >= 0
             else f'<span class="mm-badge-warn">⚠ ₹{abs(above_msp):.0f} below MSP</span>')

    st.markdown(f"""
    <div class="mm-card">
      <div class="mm-card-layer">Layer 1</div>
      <div class="mm-card-title">Price Prediction Model</div>
      <div class="mm-price-big">₹{predicted_price:,.0f}</div>
      <div class="mm-price-range">Estimated range ₹{predicted_price-MAE:,.0f} — ₹{predicted_price+MAE:,.0f} per quintal</div>
      {badge}
      <div style="margin-top:24px">
        <div class="mm-stat-row">
          <span class="mm-stat-key">Data updated up to</span>
          <span class="mm-stat-val">{last_updated}</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Algorithm</span>
          <span class="mm-stat-val">Linear Regression</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Avg error (MAE)</span>
          <span class="mm-stat-val">±₹86 / quintal</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Accuracy (MAPE)</span>
          <span class="mm-stat-val">96.4%</span>
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
        box_cls, dot_cls = "buy", "dot-green"
        label_color = "#15803D"
        label = "Buy Signal — Market Opportunity Detected"
        meta  = f"""Yesterday's price: <strong>₹{price_lag1:,.0f}</strong><br>
        3-day moving avg: ₹{price_roll3_mean:,.0f}<br>
        Market depression: ₹{abs(gap):.0f} below baseline<br>
        Historical recovery strength: <strong>67.3%</strong> within 3 days"""
    elif signal_paused:
        box_cls, dot_cls = "pause", "dot-amber"
        label_color = "#D97706"
        label = "Signal Paused This Month"
        meta  = "Feb and Apr have below 50% recovery rate.<br>VyaparIQ suppresses signals in these months to protect against false positives."
    else:
        box_cls, dot_cls = "wait", "dot-gray"
        label_color = "#4B5563"
        label = "Wait — No Opportunity Today"
        meta  = f"""Yesterday's price: ₹{price_lag1:,.0f}<br>
        3-day moving avg: ₹{price_roll3_mean:,.0f}<br>
        Gap: ₹{abs(gap):.0f} — need ₹{THRESHOLD}+ to trigger signal"""

    st.markdown(f"""
    <div class="mm-card">
      <div class="mm-card-layer">Layer 2</div>
      <div class="mm-card-title">Commercial Buying Engine</div>
      <div class="signal-box {box_cls}">
        <div class="signal-dot {dot_cls}"></div>
        <div class="signal-label" style="color:{label_color}">{label}</div>
      </div>
      <div class="signal-meta">{meta}</div>
      <div style="margin-top:18px">
        <div class="mm-stat-row">
          <span class="mm-stat-key">Active signal months</span>
          <span class="mm-stat-val">Nov, Dec, Jan, Mar, May</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Recovery probability (T+3)</span>
          <span class="mm-stat-val">67.3%</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Avg gain on signal</span>
          <span class="mm-stat-val">₹34 / quintal</span>
        </div>
        <div class="mm-stat-row">
          <span class="mm-stat-key">Trigger threshold</span>
          <span class="mm-stat-val">₹{THRESHOLD} below 3-day avg</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Price Trend Chart ─────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:32px 40px 8px'>
  <div style='font-size:18px;font-weight:700;color:#111827'>
    60-Day Market Trend — {selected_district}, {selected_variety} ({selected_grade})
  </div>
  <div style='font-size:13px;color:#6B7280;margin-top:4px'>
    Historical price charting with buy signal markers
  </div>
</div>
""", unsafe_allow_html=True)

recent  = market_df.tail(60)
sig_pts = recent[
    (recent['price_lag1'] - recent['price_roll3_mean'] < -THRESHOLD) &
    (recent['price_lag1'] > MSP) &
    (recent['date'].dt.month.isin(GOOD_MONTHS))
]

fig, ax = plt.subplots(figsize=(13, 3.8))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#FAFAFA')

ax.plot(recent['date'], recent['modal_price'],
        color='#1B4332', linewidth=2, label='Modal price')
ax.plot(recent['date'], recent['price_roll3_mean'],
        color='#D97706', linewidth=1.5, linestyle='--', label='3-day avg')
ax.axhline(MSP, color='#9CA3AF', linewidth=0.8, linestyle=':')
ax.text(recent['date'].iloc[0], MSP + 8, 'MSP ₹2425',
        fontsize=9, color='#9CA3AF')

if len(sig_pts):
    ax.scatter(sig_pts['date'], sig_pts['price_lag1'],
               color='#15803D', s=80, zorder=5,
               marker='^', label='Buy signal')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#E5E7EB')
ax.spines['bottom'].set_color('#E5E7EB')
ax.tick_params(colors='#9CA3AF', labelsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
ax.legend(loc='upper right', fontsize=10, framealpha=0, labelcolor='#4B5563')
plt.tight_layout()

st.markdown("<div style='padding:0 40px'>", unsafe_allow_html=True)
st.pyplot(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mm-footer">
  <span>VyaparIQ v1.0 — Regular wheat · MP mandis · Nov–May season · Scope: {last_updated}</span>
  <span>Data: agmarknet.gov.in · Weather: Open-Meteo</span>
</div>
""", unsafe_allow_html=True)
