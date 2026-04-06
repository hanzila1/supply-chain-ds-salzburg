"""
Supply Chain DS Dashboard
==========================
Interactive Streamlit dashboard combining:
  - Demand forecasting with external drivers
  - EUDR supplier risk scoring
  - Risk-adjusted safety stock recommendations

Run: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (
    generate_fmcg_demand, train_demand_model,
    get_supplier_portfolio, portfolio_weighted_risk,
    risk_to_disruption_prob, risk_adjusted_safety_stock
)

# ─── Page config ──────────────────────────────────────
st.set_page_config(
    page_title='Supply Chain DS — Salzburg',
    page_icon='📦',
    layout='wide'
)

st.title('📦 Supply Chain DS Dashboard')
st.caption('Demand Forecasting · EUDR Supplier Risk · Safety Stock Optimisation')
st.divider()

# ─── Sidebar controls ─────────────────────────────────
st.sidebar.header('Parameters')

forecast_horizon = st.sidebar.slider(
    'Forecast horizon (weeks)', 4, 24, 12)
service_level = st.sidebar.selectbox(
    'Service level', ['90% (z=1.28)', '95% (z=1.64)', '98% (z=2.05)', '99% (z=2.33)'],
    index=1
)
z_map = {'90% (z=1.28)': 1.28, '95% (z=1.64)': 1.64,
         '98% (z=2.05)': 2.05, '99% (z=2.33)': 2.33}
z_score = z_map[service_level]

st.sidebar.divider()
st.sidebar.subheader('EUDR Risk Override')
sugar_risk_override = st.sidebar.slider(
    'Sugar supplier risk score', 0.0, 1.0, 0.48, 0.01)
coffee_risk_override = st.sidebar.slider(
    'Coffee supplier risk score', 0.0, 1.0, 0.14, 0.01)

# ─── Tab layout ───────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    '📈 Demand Forecast',
    '🌿 EUDR Supplier Risk',
    '📊 Safety Stock'
])

# ─── Tab 1: Demand Forecast ───────────────────────────
with tab1:
    st.subheader('XGBoost Demand Forecasting with External Drivers')

    with st.spinner('Training model...'):
        df = generate_fmcg_demand()
        model, predictions, actuals, mae, features = train_demand_model(
            df, forecast_horizon=forecast_horizon)

    mape = float(np.mean(np.abs(actuals - predictions) / actuals) * 100)
    bias = float(np.mean(actuals - predictions))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('MAE', f'{mae:.0f} units/wk')
    col2.metric('MAPE', f'{mape:.1f}%')
    col3.metric('Bias', f'{bias:+.0f} units/wk')
    col4.metric('Features', len(features))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actuals, label='Actual', color='#2C2C2A', linewidth=1.8)
    ax.plot(predictions, label='XGBoost forecast', color='#185FA5',
            linestyle='--', linewidth=1.5)
    ax.fill_between(range(len(predictions)),
                    predictions * 0.9, predictions * 1.1,
                    alpha=0.15, color='#185FA5', label='±10% band')
    ax.set_title(f'{forecast_horizon}-Week Forecast Evaluation', fontsize=12)
    ax.set_xlabel('Week')
    ax.set_ylabel('Weekly Demand (units)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander('Feature importance'):
        imp = pd.Series(model.feature_importances_, index=features)
        imp = imp.sort_values(ascending=True).tail(10)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        imp.plot(kind='barh', ax=ax2, color='#185FA5', alpha=0.8)
        ax2.set_title('Top 10 Features by Importance')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ─── Tab 2: EUDR Supplier Risk ─────────────────────────
with tab2:
    st.subheader('EUDR Deforestation Risk — Supplier Portfolio')

    suppliers = get_supplier_portfolio()
    # Apply overrides from sidebar
    suppliers.loc[suppliers['commodity'] == 'sugar', 'eudr_risk'] = sugar_risk_override
    suppliers.loc[suppliers['commodity'] == 'coffee', 'eudr_risk'] = coffee_risk_override

    suppliers['risk_level'] = pd.cut(
        suppliers['eudr_risk'],
        bins=[0, 0.2, 0.4, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    suppliers['eudr_compliant'] = suppliers['eudr_risk'] < 0.4

    col1, col2, col3 = st.columns(3)
    col1.metric('Total suppliers', len(suppliers))
    col2.metric('Compliant', int(suppliers['eudr_compliant'].sum()),
                delta=f'{suppliers["eudr_compliant"].mean()*100:.0f}%')
    col3.metric('High risk', int((suppliers['risk_level'] == 'HIGH').sum()),
                delta_color='inverse',
                delta=f'{(suppliers["risk_level"]=="HIGH").mean()*100:.0f}%')

    # Risk table
    display_cols = ['supplier_id', 'country', 'commodity',
                    'volume_tons', 'eudr_risk', 'risk_level', 'eudr_compliant']

    def style_risk(val):
        if val == 'HIGH':
            return 'color: #D85A30; font-weight: bold'
        elif val == 'MEDIUM':
            return 'color: #BA7517'
        return 'color: #1D9E75'

    st.dataframe(
        suppliers[display_cols].style.applymap(
            style_risk, subset=['risk_level']),
        use_container_width=True, hide_index=True
    )

    # Risk bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    color_map = {'LOW': '#1D9E75', 'MEDIUM': '#EF9F27', 'HIGH': '#D85A30'}
    colors = [color_map[r] for r in suppliers['risk_level']]
    ax3.barh(suppliers['supplier_id'], suppliers['eudr_risk'],
             color=colors, edgecolor='white', height=0.6)
    ax3.axvline(0.2, color='#1D9E75', linestyle='--', alpha=0.6, label='LOW threshold')
    ax3.axvline(0.4, color='#D85A30', linestyle='--', alpha=0.6, label='HIGH threshold')
    ax3.set_xlabel('EUDR Deforestation Risk Score')
    ax3.set_title('Supplier Risk Portfolio', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.info('🛰️ In production: risk scores are computed from Google Earth Engine '
            '(Hansen Global Forest Change dataset) using GPS coordinates from '
            'your procurement system.')

# ─── Tab 3: Safety Stock ───────────────────────────────
with tab3:
    st.subheader('Risk-Adjusted Safety Stock Recommendations')

    st.markdown("""
    Safety stock is calculated as:
    ```
    SS = (demand variability buffer) + (EUDR supply disruption buffer)
    ```
    The disruption buffer increases as supplier EUDR risk increases —
    connecting sustainability compliance directly to inventory planning.
    """)

    coffee_risk = portfolio_weighted_risk(suppliers, 'coffee')
    sugar_risk  = portfolio_weighted_risk(suppliers, 'sugar')
    coffee_disr = risk_to_disruption_prob(coffee_risk)
    sugar_disr  = risk_to_disruption_prob(sugar_risk)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**☕ Coffee Ingredient**')
        coffee_ss = risk_adjusted_safety_stock(
            180, 25, 8, z_score, coffee_disr)
        st.metric('Total safety stock', f'{coffee_ss["total"]:.0f} tons/week')
        st.metric('Disruption buffer', f'{coffee_ss["ss_disruption"]:.0f} tons',
                  help='Additional stock due to EUDR supply risk')
        st.metric('Disruption probability', f'{coffee_disr*100:.1f}% /year')

    with col2:
        st.markdown('**🍬 Sugar Ingredient**')
        sugar_ss = risk_adjusted_safety_stock(
            500, 60, 4, z_score, sugar_disr)
        st.metric('Total safety stock', f'{sugar_ss["total"]:.0f} tons/week')
        st.metric('Disruption buffer', f'{sugar_ss["ss_disruption"]:.0f} tons',
                  help='Additional stock due to EUDR supply risk')
        st.metric('Disruption probability', f'{sugar_disr*100:.1f}% /year')

    # Scenario comparison chart
    st.divider()
    st.markdown('**📉 Safety Stock Sensitivity to Supplier Risk**')

    risk_range = np.linspace(0, 0.8, 50)
    ss_totals = [
        risk_adjusted_safety_stock(
            500, 60, 4, z_score,
            risk_to_disruption_prob(r))['total']
        for r in risk_range
    ]

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(risk_range, ss_totals, color='#185FA5', linewidth=2)
    ax4.axvline(sugar_risk, color='#D85A30', linestyle='--',
                label=f'Current sugar risk: {sugar_risk:.2f}')
    ax4.axvline(coffee_risk, color='#1D9E75', linestyle='--',
                label=f'Current coffee risk: {coffee_risk:.2f}')
    ax4.set_xlabel('EUDR Risk Score')
    ax4.set_ylabel('Required Safety Stock (tons/week)')
    ax4.set_title(
        'How EUDR Compliance Improvement Reduces Safety Stock Requirements',
        fontsize=12)
    ax4.legend()
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.success(
        '💡 Use the sidebar sliders to simulate the impact of switching to '
        'lower-risk suppliers or achieving EUDR certification.'
    )
