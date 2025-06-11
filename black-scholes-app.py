import streamlit as st
import numpy as np
from scipy.stats import norm # for the cumulative distribution function (CDF)
import matplotlib.pyplot as plt
import pandas as pd # for easier data handling for charts
import seaborn as sns
from fpdf import FPDF
import tempfile


# --- Core Black-Scholes and Greeks Algorithms ---

#############################
# Core formula functions
#############################
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    initial_sidebar_state="expanded",
    layout="wide")

def normal_cdf(x):
    return norm.cdf(x) # calculates cumulative distribution func of the standard normal distribution

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option Type must be 'call' or 'put'")
    return price # returns the black-scholes price for the option

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
         raise ValueError("Option Type must be 'call' or 'put'")
    return delta # returns the option delta (sensitivity to underlying price)

def calculate_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma # returns the rate of change of delta with respect to the underlying asset price (gamma)

def calculate_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pdf_d1 = norm.pdf(d1)
    vega = S * pdf_d1 * np.sqrt(T)
    return vega / 100 # return vega per 1% change

def calculate_theta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("Option Type must be 'call' or 'put'")
    return theta / 365 # return per day theta

def calculate_rho(S, K, T, r, sigma, option_type='call'):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * normal_cdf(d2)
    elif option_type == 'put':
        rho = -K * T * np.exp(-r * T) * normal_cdf(-d2)
    else:
         raise ValueError("Option Type must be 'call' or 'put'")

    return rho / 100 # returns as a percentage

def display_input_summary(S, K, T, sigma, r):
    input_df = pd.DataFrame({
        "Current Asset Price": [S],
        "Strike Price": [K],
        "Time to Maturity (Years)": [T],
        "Volatility (σ)": [sigma],
        "Risk-Free Interest Rate": [r]
    })
    st.dataframe(input_df.style.format("{:.4f}"), use_container_width=True)

def display_option_value_cards(S, K, T, r, sigma):
    call_value = black_scholes_price(S, K, T, r, sigma, "call")
    put_value = black_scholes_price(S, K, T, r, sigma, "put")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#7fdc8c;padding:10px 0 10px 0;border-radius:15px;text-align:center;">
                <span style="font-size:16px;font-weight:bold;color:black;">CALL Value</span><br>
                <span style="font-size:22px;font-weight:bold;color:black;">${call_value:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background-color:#ffb3b3;padding:10px 0 10px 0;border-radius:15px;text-align:center;">
                <span style="font-size:16px;font-weight:bold;color:black;">PUT Value</span><br>
                <span style="font-size:22px;font-weight:bold;color:black;">${put_value:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

def plot_bs_heatmaps(spot_range, vol_range, K, T, r):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            call_prices[i, j] = black_scholes_price(spot, K, T, r, vol, "call")
            put_prices[i, j] = black_scholes_price(spot, K, T, r, vol, "put")
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap=cmap, ax=ax_call, cbar_kws={'label': 'Call Price'})
    ax_call.set_title('CALL Option Price Heatmap')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap=cmap, ax=ax_put, cbar_kws={'label': 'Put Price'})
    ax_put.set_title('PUT Option Price Heatmap')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    return fig_call, fig_put

###############################
# UI (sidebar)
##############################
st.title("📊 Black-Scholes Model")

with st.expander("What is the Black-Scholes Model?"):
    st.markdown("""
    The Black-Scholes model is a mathematical model for pricing European options and derivatives. 
    It estimates the theoretical value of options based on factors like spot price, strike price, time to maturity, volatility, and risk-free rate.
    """)

with st.sidebar: # used for simplicity 
    st.markdown("""
    ### Created by: Jordan Buckley
    """)
    st.markdown("""
    <a href="https://www.linkedin.com/in/jordan05/" target="_blank" style="text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/24/000000/linkedin.png"/> LinkedIn
    </a>
    <a href="https://github.com/JordanBuckleyGit" target="_blank" style="text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/24/000000/github.png"/> GitHub
    </a>
    <a href="mailto:jordanbuckleycork@gmail.com" target="_blank" style="text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/24/000000/new-post.png"/> Contact
    </a>                   
    """, unsafe_allow_html=True)
    st.markdown("---")
    S = st.number_input(
        "Spot Price (S)", 
        min_value=1.0, 
        max_value=500.0, 
        value=100.0, 
        step=1.0,
        help="The current price of the underlying asset."
    )
    K = st.number_input(
        "Strike price (K)", 
        min_value=1.0, 
        max_value=500.0, 
        value=100.0, 
        step=1.0,
        help="The price at which you have the right to buy (call) or sell (put) the asset."
    )
    T = st.number_input(
        "Time to Maturity (Years, T)", 
        min_value=0.01, 
        max_value=5.0, 
        value=1.0, 
        step=0.01,
        help="The time in years until the option expires."
    )
    r = st.number_input(
        "Risk-Free Rate (r, decimal)", 
        min_value=0.0, 
        max_value=0.2, 
        value=0.05, 
        step=0.001,
        help="The annualized risk-free interest rate (as a decimal, e.g., 0.05 for 5%)."
    )
    sigma = st.number_input(
        "Volatility (σ, decimal)", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.2, 
        step=0.01,
        help="The annualized volatility of the underlying asset (as a decimal, e.g., 0.2 for 20%)."
    )
    option_type = st.selectbox(
        "Option Type", 
        ["call", "put"],
        help="Choose 'call' for the right to buy, or 'put' for the right to sell."
    )
    st.markdown("---")
    st.subheader("PNL")
    option_price = st.number_input(
        "Option Price (Premium)", 
        min_value=0.0, 
        value=float(f"{black_scholes_price(S, K, T, r, sigma, option_type):.2f}"), 
        step=0.01,
        help="The price paid per option contract (premium)."
    )
    num_contracts = st.number_input(
        "Number of Option Contracts", 
        min_value=1, 
        value=1, 
        step=1,
        help="The number of option contracts you want to buy or sell."
    )
    calculate_button = st.button(
        "Calculate",
        help="Click to calculate the P&L and payoff chart."
    )
    st.markdown("---")
    calculate_btn = st.button(
        "Heatmap Parameters",
        help="Click to update the heatmap parameter ranges below."
    )
    spot_min = st.number_input(
        "Min Spot Price", 
        min_value=0.01, 
        value=S*0.8, 
        step=0.01,
        help="The minimum spot price to display on the heatmap."
    )
    spot_max = st.number_input(
        "Max Spot Price", 
        min_value=0.01, 
        value=S*1.2, 
        step=0.01,
        help="The maximum spot price to display on the heatmap."
    )
    vol_min = st.slider(
        "Min Volatility for Heatmap", 
        min_value=0.01, 
        max_value=1.0, 
        value=sigma*0.5, 
        step=0.01,
        help="The minimum volatility value to display on the heatmap."
    )
    vol_max = st.slider(
        "Max Volatility for Heatmap", 
        min_value=0.01, 
        max_value=1.0, 
        value=sigma*1.5, 
        step=0.01,
        help="The maximum volatility value to display on the heatmap."
    )
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

display_input_summary(S, K, T, sigma, r)
display_option_value_cards(S, K, T, r, sigma)

####################################
# Graphing Section
####################################
st.header("Options Price - Heatmaps")
st.info("These heatmaps show how the theoretical price of call and put options changes as spot price and volatility vary. Use the sliders in the sidebar to explore different scenarios.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Price Heatmap")
    fig_call, fig_put = plot_bs_heatmaps(spot_range, vol_range, K, T, r)
    st.pyplot(fig_call)
with col2:
    st.subheader("Put Price Heatmap")
    st.pyplot(fig_put)
st.caption("Tip: Try increasing volatility to see how it impacts both call and put prices!")

# --- Option Payoff at Expiry (P&L) ---
st.header("Option Payoff at Expiry (PNL)")
S_range = np.linspace(max(1.0, S - 50), S + 50, 100)
contract_size = 100  # standard contract size for options

if option_type == "call":
    payoff = np.maximum(S_range - K, 0) - option_price
    bep = K + option_price
else:
    payoff = np.maximum(K - S_range, 0) - option_price
    bep = K - option_price

total_payoff = payoff * num_contracts * contract_size

fig_payoff, ax_payoff = plt.subplots(figsize=(10, 6))

ax_payoff.fill_between(S_range, total_payoff, 0, where=(S_range >= bep), color='green', alpha=0.15)
ax_payoff.fill_between(S_range, total_payoff, 0, where=(S_range < bep), color='red', alpha=0.15)

ax_payoff.plot(S_range, total_payoff, label=f'{option_type.capitalize()} Option P&L', color='royalblue', linewidth=2)
ax_payoff.axhline(0, color='black', linestyle='--', linewidth=1)
ax_payoff.axvline(K, color='red', linestyle=':', linewidth=1, label='Strike Price')

ax_payoff.text(
    0.02, 0.98,
    f"Break-even Point (BEP): €{bep:.2f}",
    transform=ax_payoff.transAxes,
    fontsize=12,
    fontweight='bold',
    color='green',
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='green')
)

ax_payoff.set_title(f"{option_type.capitalize()} Option Payoff at Expiry")
ax_payoff.set_xlabel("Spot Price at Expiry (S)")
ax_payoff.set_ylabel("Profit / Loss (€)")
ax_payoff.grid(True, linestyle='--', alpha=0.7)
ax_payoff.legend()
st.pyplot(fig_payoff)

contract_size = 100  # standard contract size for options
total_cost = option_price * num_contracts * contract_size
intrinsic_value = max(0, S - K) if option_type == "call" else max(0, K - S)
potential_profit = (intrinsic_value - option_price) * num_contracts * contract_size
potential_return = (potential_profit / total_cost * 100) if total_cost > 0 else 0

with st.expander("See Example P&L Scenarios"):
    col_call, col_put = st.columns(2)
    with col_call:
        st.markdown("""
        ### Call Option Examples
        **Example 1: Profit**
        - Spot Price at Expiry: **€120**
        - Strike Price: **€100**
        - Option Type: **Call**
        - Option Premium: **€5**
        - Number of Contracts: **1**
        - Contract Size: **100**
        - **Calculation:**  
          Intrinsic Value = max(120 - 100, 0) = €20  
          Profit = (Intrinsic Value - Premium) × Contracts × Contract Size  
          Profit = (20 - 5) × 1 × 100 = **€1,500**

        **Example 2: Loss**
        - Spot Price at Expiry: **€90**
        - Strike Price: **€100**
        - Option Type: **Call**
        - Option Premium: **€5**
        - Number of Contracts: **1**
        - Contract Size: **100**
        - **Calculation:**  
          Intrinsic Value = max(90 - 100, 0) = €0  
          Loss = (Intrinsic Value - Premium) × Contracts × Contract Size  
          Loss = (0 - 5) × 1 × 100 = **-€500**

        **Example 3: Neither (Break-even)**
        - Spot Price at Expiry: **€105**
        - Strike Price: **€100**
        - Option Type: **Call**
        - Option Premium: **€5**
        - Number of Contracts: **1**
        - Contract Size: **100**
        - **Calculation:**  
          Intrinsic Value = max(105 - 100, 0) = €5  
          P&L = (5 - 5) × 1 × 100 = **€0** (Break-even)
        """)
    with col_put:
        st.markdown("""
        ### Put Option Examples
        **Example 1: Profit**
        - Spot Price at Expiry: **€80**
        - Strike Price: **€100**
        - Option Type: **Put**
        - Option Premium: **€5**
        - Number of Contracts: **1**
        - Contract Size: **100**
        - **Calculation:**  
          Intrinsic Value = max(100 - 80, 0) = €20  
          Profit = (Intrinsic Value - Premium) × Contracts × Contract Size  
          Profit = (20 - 5) × 1 × 100 = **€1,500**

        **Example 2: Loss**
        - Spot Price at Expiry: **€110**
        - Strike Price: **€100**
        - Option Type: **Put**
        - Option Premium: **€5**
        - Number of Contracts: **1**
        - Contract Size: **100**
        - **Calculation:**  
          Intrinsic Value = max(100 - 110, 0) = €0  
          Loss = (Intrinsic Value - Premium) × Contracts × Contract Size  
          Loss = (0 - 5) × 1 × 100 = **-€500**

        **Example 3: Neither (Break-even)**
        - Spot Price at Expiry: **€95**
        - Strike Price: **€100**
        - Option Type: **Put**
        - Option Premium: **€5**
        - Number of Contracts: **1**
        - Contract Size: **100**
        - **Calculation:**  
          Intrinsic Value = max(100 - 95, 0) = €5  
          P&L = (5 - 5) × 1 × 100 = **€0** (Break-even)
        """)
    
st.subheader("Profit & Loss (P&L)")
col1, col2, col3 = st.columns(3)
col1.metric("Total Option Cost (Premium)", f"€ {total_cost:,.2f}")
col2.metric("Potential Profit", f"€ {potential_profit:,.2f}")
col3.metric("Potential Return", f"{potential_return:.2f} %")

###########################
# Output Message
###########################
if option_type == "call":
    in_the_money = S > K
    net_profit = (max(0, S - K) - option_price) * num_contracts * contract_size
elif option_type == "put":
    in_the_money = S < K
    net_profit = (max(0, K - S) - option_price) * num_contracts * contract_size

if in_the_money and net_profit > 0:
    st.success("This option is **in the money** and profitable at expiry.")
elif in_the_money and net_profit <= 0:
    st.warning("This option is **in the money** but **not profitable** (the gain does not cover the premium).")
elif not in_the_money and net_profit < 0:
    st.info("This option is **out of the money** and not profitable.")
else:
    st.info("This option is **at the money** (spot price equals strike price).")

#############################
# Faqs (kinda)
#############################
with st.expander("Information"):
    st.markdown("""
    - **Spot Price (S):** The current price of the underlying asset.
    - **Strike Price (K):** The price at which you can buy (call) or sell (put) the asset.
    - **Volatility (σ):** A measure of how much the asset price fluctuates.
    - **Risk-Free Rate (r):** Theoretical return of an investment with zero risk.
    """)
st.markdown('[Learn more about Black-Scholes on Investopedia](https://www.investopedia.com/terms/b/blackscholes.asp)')

################################
# PDF summary
################################
def create_pdf(S, K, T, sigma, r, option_type, option_price, num_contracts, total_cost, potential_profit, potential_return, fig_payoff, fig_call, fig_put):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Black-Scholes Option Pricing Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Spot Price (S): {S}", ln=True)
    pdf.cell(200, 10, txt=f"Strike Price (K): {K}", ln=True)
    pdf.cell(200, 10, txt=f"Time to Maturity (T): {T}", ln=True)
    pdf.cell(200, 10, txt=f"Volatility (sigma): {sigma}", ln=True)
    pdf.cell(200, 10, txt=f"Risk-Free Rate (r): {r}", ln=True)
    pdf.cell(200, 10, txt=f"Option Type: {option_type}", ln=True)
    pdf.cell(200, 10, txt=f"Option Price (Premium): {option_price} EUR", ln=True)
    pdf.cell(200, 10, txt=f"Number of Contracts: {num_contracts}", ln=True)
    pdf.cell(200, 10, txt=f"Total Cost: {total_cost} EUR", ln=True)
    pdf.cell(200, 10, txt=f"Potential Profit: {potential_profit} EUR", ln=True)
    pdf.cell(200, 10, txt=f"Potential Return: {potential_return:.2f}%", ln=True)
    pdf.ln(10)

    def add_fig_to_pdf(fig, pdf, title):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight')
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt=title, ln=True, align='C')
            pdf.image(tmpfile.name, x=10, y=30, w=pdf.w - 20)
    
    add_fig_to_pdf(fig_payoff, pdf, "Option Payoff at Expiry")
    add_fig_to_pdf(fig_call, pdf, "Call Option Price Heatmap")
    add_fig_to_pdf(fig_put, pdf, "Put Option Price Heatmap")

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

pdf_file = create_pdf(
    S, K, T, sigma, r, option_type, option_price, num_contracts, total_cost, potential_profit, potential_return,
    fig_payoff, fig_call, fig_put
)
st.download_button(
    label="Download PDF Report",
    data=pdf_file,
    file_name="black_scholes_report.pdf",
    mime="application/pdf" # tells browser its a pdf file
)