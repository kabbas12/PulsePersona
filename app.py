"""
PulsePersona: AI-Driven Financial Personalization Engine
A Streamlit application for FinTech capstone project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from faker import Faker
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="PulsePersona - AI Financial Personalization",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BETTER UI
# ============================================
st.markdown("""
<style>
    .stButton button {
        background-color: #4A90E2;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton button:hover {
        background-color: #357ABD;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZATION
# ============================================
@st.cache_resource
def initialize_faker():
    return Faker()

@st.cache_data
def generate_synthetic_data(num_users=100, transactions_per_user=100):
    """
    Generate synthetic user and transaction data
    """
    fake = Faker()
    random.seed(42)
    np.random.seed(42)
    
    START_DATE = datetime(2024, 1, 1)
    END_DATE = datetime(2024, 12, 31)
    
    # Categories
    CATEGORIES = {
        'Groceries': ['Walmart', 'Target', 'Kroger', 'Whole Foods', 'Aldi', 'Costco'],
        'Dining': ['Starbucks', 'McDonalds', 'Chipotle', 'Dominos', 'Olive Garden', 'Local Cafe'],
        'Entertainment': ['Netflix', 'Spotify', 'Amazon Prime', 'Hulu', 'Disney+', 'Movie Theater'],
        'Shopping': ['Amazon', 'eBay', 'Best Buy', 'Home Depot', "Macy's", 'Nike'],
        'Transport': ['Uber', 'Lyft', 'Shell Gas', 'Exxon', 'Parking Meter', 'Subway'],
        'Bills': ['Electric Bill', 'Water Bill', 'Internet', 'Phone Bill', 'Rent', 'Mortgage'],
        'Health': ['CVS', 'Walgreens', 'Gym Membership', 'Doctor Visit', 'Pharmacy'],
        'Travel': ['Delta Air', 'American Air', 'Airbnb', 'Marriott', 'Expedia', 'Booking.com'],
        'Investment': ['Vanguard', 'Fidelity', 'Robinhood', 'Coinbase', 'Wealthfront'],
        'Income': ['Salary Deposit', 'Freelance Payment', 'Transfer from Savings', 'Bonus']
    }
    
    # Persona spending patterns
    PERSONA_PATTERNS = {
        'Saver': {
            'Groceries': 0.8, 'Dining': 0.4, 'Entertainment': 0.3,
            'Shopping': 0.3, 'Transport': 0.5, 'Bills': 0.7,
            'Health': 0.6, 'Travel': 0.2, 'Investment': 1.5, 'Income': 1.0
        },
        'Spender': {
            'Groceries': 1.2, 'Dining': 1.8, 'Entertainment': 1.5,
            'Shopping': 2.0, 'Transport': 1.3, 'Bills': 1.0,
            'Health': 0.9, 'Travel': 1.2, 'Investment': 0.2, 'Income': 1.0
        },
        'Investor': {
            'Groceries': 0.9, 'Dining': 0.8, 'Entertainment': 0.7,
            'Shopping': 0.6, 'Transport': 0.8, 'Bills': 0.9,
            'Health': 0.8, 'Travel': 1.1, 'Investment': 2.5, 'Income': 1.3
        },
        'Debtor': {
            'Groceries': 1.1, 'Dining': 1.2, 'Entertainment': 1.1,
            'Shopping': 1.4, 'Transport': 1.1, 'Bills': 1.5,
            'Health': 1.2, 'Travel': 0.5, 'Investment': 0.1, 'Income': 0.7
        },
        'Balanced': {
            'Groceries': 1.0, 'Dining': 1.0, 'Entertainment': 1.0,
            'Shopping': 1.0, 'Transport': 1.0, 'Bills': 1.0,
            'Health': 1.0, 'Travel': 1.0, 'Investment': 1.0, 'Income': 1.0
        }
    }
    
    # Generate users
    users = []
    
    for user_id in range(1, num_users + 1):
        if user_id <= 25:
            persona = 'Saver'
        elif user_id <= 45:
            persona = 'Spender'
        elif user_id <= 60:
            persona = 'Investor'
        elif user_id <= 75:
            persona = 'Debtor'
        else:
            persona = 'Balanced'
        
        if persona == 'Investor':
            monthly_income = random.randint(6000, 12000)
        elif persona == 'Saver':
            monthly_income = random.randint(4000, 8000)
        elif persona == 'Spender':
            monthly_income = random.randint(3500, 7000)
        elif persona == 'Debtor':
            monthly_income = random.randint(2500, 4500)
        else:
            monthly_income = random.randint(4000, 9000)
        
        users.append({
            'user_id': user_id,
            'persona': persona,
            'monthly_income': monthly_income,
            'age': random.randint(22, 65),
            'city': fake.city(),
            'occupation': fake.job()
        })
    
    users_df = pd.DataFrame(users)
    
    # Generate transactions
    transactions = []
    
    for _, user in users_df.iterrows():
        user_id = user['user_id']
        persona = user['persona']
        monthly_income = user['monthly_income']
        pattern = PERSONA_PATTERNS[persona]
        
        baseline_spending = {
            'Groceries': 400, 'Dining': 200, 'Entertainment': 100,
            'Shopping': 300, 'Transport': 150, 'Bills': 1000,
            'Health': 150, 'Travel': 200, 'Investment': 300, 'Income': monthly_income
        }
        
        for _ in range(transactions_per_user):
            categories = list(CATEGORIES.keys())
            weights = [pattern[cat] for cat in categories]
            category = random.choices(categories, weights=weights)[0]
            
            if category == 'Income':
                continue
            
            base_amount = baseline_spending[category] / 10
            adjusted_amount = base_amount * pattern[category]
            amount = round(np.random.gamma(shape=2, scale=adjusted_amount/2), 2)
            amount = min(amount, baseline_spending[category] * 0.8)
            
            if category == 'Income':
                amount = abs(amount)
            else:
                amount = -abs(amount)
            
            date = START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days))
            merchant = random.choice(CATEGORIES[category])
            
            transactions.append({
                'user_id': user_id,
                'date': date.strftime('%Y-%m-%d'),
                'amount': amount,
                'category': category,
                'merchant': merchant,
                'description': f"{merchant} - {fake.bs()[:30]}"
            })
        
        # Add monthly income
        for month in range(1, 13):
            income_date = datetime(2024, month, random.randint(25, 28))
            if income_date <= END_DATE:
                transactions.append({
                    'user_id': user_id,
                    'date': income_date.strftime('%Y-%m-%d'),
                    'amount': monthly_income,
                    'category': 'Income',
                    'merchant': 'Employer',
                    'description': f'Salary Deposit - {fake.company()}'
                })
    
    transactions_df = pd.DataFrame(transactions)
    
    return users_df, transactions_df


@st.cache_data
def calculate_user_features(users_df, transactions_df):
    """
    Calculate behavioral features for each user
    """
    features_list = []
    
    for user_id in users_df['user_id'].unique():
        user_tx = transactions_df[transactions_df['user_id'] == user_id]
        
        spending_by_cat = user_tx[user_tx['amount'] < 0].groupby('category')['amount'].sum().abs()
        total_spending = spending_by_cat.sum()
        
        income = user_tx[user_tx['amount'] > 0]['amount'].sum()
        savings_rate = (income - total_spending) / income if income > 0 else 0
        
        tx_count = len(user_tx)
        avg_tx_size = user_tx['amount'].abs().mean()
        
        features = {
            'user_id': user_id,
            'total_spending': total_spending,
            'savings_rate': savings_rate,
            'tx_count': tx_count,
            'avg_tx_size': avg_tx_size
        }
        
        for cat in ['Groceries', 'Dining', 'Entertainment', 'Shopping', 'Transport', 'Bills', 'Health', 'Travel', 'Investment']:
            cat_spend = spending_by_cat.get(cat, 0)
            features[f'pct_{cat}'] = cat_spend / total_spending if total_spending > 0 else 0
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    return features_df


@st.cache_data
def perform_clustering(features_df):
    """
    Perform K-Means clustering to identify personas
    """
    cluster_features = ['savings_rate', 'total_spending', 'tx_count', 'avg_tx_size']
    X = features_df[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to persona names
    cluster_persona_map = {}
    for cluster in range(4):
        cluster_data = features_df[features_df['cluster'] == cluster]
        avg_savings = cluster_data['savings_rate'].mean()
        if avg_savings > 0.3:
            persona = 'Saver'
        elif avg_savings < 0:
            persona = 'Debtor'
        elif cluster_data['total_spending'].mean() > features_df['total_spending'].median():
            persona = 'Spender'
        else:
            persona = 'Balanced'
        cluster_persona_map[cluster] = persona
    
    features_df['persona'] = features_df['cluster'].map(cluster_persona_map)
    return features_df, kmeans, scaler


def generate_nudges(user_features, persona, user_transactions):
    """
    Generate personalized nudges based on user persona and behavior
    """
    nudges = []
    
    if persona == 'Saver':
        if user_features['savings_rate'] > 0.4:
            nudges.append({
                'icon': '✅',
                'title': 'Excellent Savings Rate!',
                'message': "You're saving over 40% of your income! Consider moving excess cash to a high-yield savings account earning 4%+ APY.",
                'action': 'Compare HYSA rates →'
            })
        else:
            nudges.append({
                'icon': '💰',
                'title': 'Automate Your Savings',
                'message': "You're a natural saver. Try automating $50/week to an investment account to make your money work harder.",
                'action': 'Set up auto-invest →'
            })
        nudges.append({
            'icon': '📈',
            'title': 'Round-Up Feature',
            'message': "Link your debit card to round purchases to the nearest dollar and invest the difference. The average user saves $45/month.",
            'action': 'Enable round-ups →'
        })
    
    elif persona == 'Spender':
        dining_pct = user_features.get('pct_Dining', 0)
        if dining_pct > 0.2:
            nudges.append({
                'icon': '🍕',
                'title': 'High Dining Spend Detected',
                'message': f"You spend {dining_pct*100:.1f}% of your budget on dining out. Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
                'action': 'View budget template →'
            })
        nudges.append({
            'icon': '💳',
            'title': 'Maximize Credit Card Rewards',
            'message': "Switch to a 2% cash-back card for all purchases. Based on your spending, that's ~$400 back annually.",
            'action': 'Find best card →'
        })
        nudges.append({
            'icon': '⏰',
            'title': '24-Hour Rule',
            'message': "Before any purchase over $100, wait 24 hours. Users who follow this skip 30% of impulse buys.",
            'action': 'Enable purchase reminders →'
        })
    
    elif persona == 'Investor':
        nudges.append({
            'icon': '📊',
            'title': 'Portfolio Rebalancing',
            'message': "Your estimated portfolio returned 8.2% this year. Rebalance to maintain your target 70/30 stock/bond ratio.",
            'action': 'Rebalance now →'
        })
        nudges.append({
            'icon': '💡',
            'title': 'Tax-Loss Harvesting',
            'message': "You have unrealized losses of ~$1,200. Selling these could offset capital gains and save you ~$250 in taxes.",
            'action': 'Review opportunities →'
        })
        nudges.append({
            'icon': '🎯',
            'title': 'Goal Progress',
            'message': "You're 65% of the way to your retirement goal. Increase contributions by 2% to reach it 3 years earlier.",
            'action': 'Adjust contributions →'
        })
    
    elif persona == 'Debtor':
        nudges.append({
            'icon': '⚠️',
            'title': 'Debt Reduction Strategy',
            'message': "Your debt-to-income ratio is high. Try the 'avalanche method': Pay minimum on all debts, put extra toward highest interest card.",
            'action': 'Create debt payoff plan →'
        })
        nudges.append({
            'icon': '🔄',
            'title': 'Balance Transfer Offer',
            'message': "You may qualify for 0% APR for 18 months on balance transfers. Could save you ~$1,200 in interest.",
            'action': 'Check eligibility →'
        })
        nudges.append({
            'icon': '📉',
            'title': 'Lower Your Interest Rate',
            'message': "One 5-minute call to your credit card company could lower your APR by 3-5%, saving $300+ per year.",
            'action': 'Get negotiation script →'
        })
    
    else:  # Balanced
        nudges.append({
            'icon': '⚖️',
            'title': 'Optimize Your Balance',
            'message': "You're maintaining a healthy financial balance. Consider increasing investments by 5% of your income.",
            'action': 'Review portfolio →'
        })
        nudges.append({
            'icon': '🎯',
            'title': 'One Small Change',
            'message': "Skip one delivery order per week. Save $200/month → $2,400/year → $38,000 over 10 years (invested).",
            'action': 'Track delivery spending →'
        })
        nudges.append({
            'icon': '📈',
            'title': 'Level Up Your Savings',
            'message': "Increase your 401(k) contribution by 1%. Your employer matches 50% — that's free money!",
            'action': 'Update 401(k) →'
        })
    
    return nudges[:3]


def monte_carlo_projection(current_savings, monthly_savings, years=10, return_rate=0.07, volatility=0.15, n_simulations=500):
    """
    Monte Carlo simulation for wealth projection
    """
    months = years * 12
    results = []
    
    for _ in range(n_simulations):
        portfolio = current_savings
        monthly_returns = np.random.normal(return_rate/12, volatility/np.sqrt(12), months)
        
        for month_return in monthly_returns:
            portfolio = portfolio * (1 + month_return) + monthly_savings
        
        results.append(portfolio)
    
    return {
        'median': np.median(results),
        'mean': np.mean(results),
        'p10': np.percentile(results, 10),
        'p25': np.percentile(results, 25),
        'p75': np.percentile(results, 75),
        'p90': np.percentile(results, 90),
        'all_results': results
    }


def get_persona_color(persona):
    """Return color for each persona"""
    colors = {
        'Saver': '#2E86AB',
        'Spender': '#F18F01',
        'Investor': '#2D6A4F',
        'Debtor': '#D62828',
        'Balanced': '#6B4E71'
    }
    return colors.get(persona, '#4A90E2')


def get_persona_icon(persona):
    """Return icon for each persona"""
    icons = {
        'Saver': '🐷',
        'Spender': '💳',
        'Investor': '📈',
        'Debtor': '⚠️',
        'Balanced': '⚖️'
    }
    return icons.get(persona, '💜')


def get_persona_description(persona):
    """Return description for each persona"""
    descriptions = {
        'Saver': "You're a natural saver who prioritizes financial security. You live below your means and consistently put money aside for the future.",
        'Spender': "You enjoy life's pleasures and aren't afraid to spend on experiences. The key is balancing enjoyment with future goals.",
        'Investor': "You understand that money should work for you. You're focused on building long-term wealth through strategic investments.",
        'Debtor': "You're currently carrying debt that's impacting your financial flexibility. The good news is there are clear paths to freedom.",
        'Balanced': "You've found a healthy equilibrium between spending today and saving for tomorrow. Keep optimizing!"
    }
    return descriptions.get(persona, "A unique financial personality with room to grow.")


def create_spending_pie_chart(spending_by_cat, persona_color):
    """Create a matplotlib pie chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(spending_by_cat)))
    wedges, texts, autotexts = ax.pie(
        spending_by_cat.values,
        labels=spending_by_cat.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax.set_title('Your Spending by Category', fontsize=14, fontweight='bold')
    return fig


def create_monthly_cashflow_chart(monthly_income, monthly_spending, persona_color):
    """Create a matplotlib bar chart for cashflow"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    months = range(len(monthly_income))
    width = 0.35
    
    ax.bar([m - width/2 for m in months], monthly_income.values, width, 
           label='Income', color='#2E86AB', alpha=0.8)
    ax.bar([m + width/2 for m in months], monthly_spending.values, width, 
           label='Spending', color='#F18F01', alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Amount ($)', fontsize=12)
    ax.set_title('Monthly Income vs Spending', fontsize=14, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels([str(m) for m in range(1, len(months)+1)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def create_monte_carlo_chart(years, medians, p10s, p90s, persona_color):
    """Create a matplotlib line chart with confidence bands"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(years, medians, 'b-', linewidth=2, label='Median', color=persona_color)
    ax.fill_between(years, p10s, p90s, alpha=0.3, label='10th-90th Percentile', color=persona_color)
    
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Wealth Projection (Monte Carlo Simulation)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with dollar signs
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    return fig


# ============================================
# MAIN APP
# ============================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("💜 PulsePersona")
        st.markdown("### AI-Powered Financial Companion")
        st.markdown("---")
        
        # Data generation
        st.subheader("📊 Data Settings")
        num_users = st.slider("Number of Users", 50, 200, 100, 25)
        transactions_per_user = st.slider("Transactions per User", 50, 150, 100, 25)
        
        if st.button("🔄 Generate New Data", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        st.subheader("📚 Course Alignment")
        st.caption("✅ LO1: FinTech & AI Role")
        st.caption("✅ LO2: ML Techniques (Clustering)")
        st.caption("✅ LO3: Strategic Analysis")
        st.caption("✅ LO4: Ethics & Privacy")
        st.caption("✅ LO5: Communication")
        st.caption("✅ LO6: Team Collaboration")
        
        st.markdown("---")
        st.caption("PulsePersona v1.0 | Capstone Project")
    
    # Main content
    st.title("💜 PulsePersona")
    st.markdown("### Your AI-Powered Financial Companion")
    st.markdown("*Personalized insights, nudges, and projections based on your unique financial behavior*")
    st.markdown("---")
    
    # Load or generate data
    with st.spinner("Loading financial data..."):
        users_df, transactions_df = generate_synthetic_data(num_users, transactions_per_user)
        features_df = calculate_user_features(users_df, transactions_df)
        features_df, kmeans, scaler = perform_clustering(features_df)
    
    # User selection
    user_ids = sorted(users_df['user_id'].unique())
    selected_user = st.selectbox("Select a User", user_ids, format_func=lambda x: f"👤 User {x}")
    
    # Get user data
    user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
    user_features = features_df[features_df['user_id'] == selected_user].iloc[0]
    user_transactions = transactions_df[transactions_df['user_id'] == selected_user]
    
    persona = user_features['persona']
    persona_color = get_persona_color(persona)
    persona_icon = get_persona_icon(persona)
    
    # Calculate financial metrics
    income_total = user_transactions[user_transactions['amount'] > 0]['amount'].sum()
    spending_total = user_transactions[user_transactions['amount'] < 0]['amount'].abs().sum()
    current_savings = income_total - spending_total
    monthly_savings = current_savings / 12 if current_savings > 0 else 50
    savings_rate = (current_savings / income_total * 100) if income_total > 0 else 0
    
    # Persona header
    st.markdown(f"""
    <div style='background-color: {persona_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {persona_color}; margin-bottom: 20px;'>
        <h2 style='margin: 0;'>{persona_icon} {persona} Persona</h2>
        <p style='margin: 10px 0 0 0;'>{get_persona_description(persona)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Income", f"${income_total:,.0f}")
    with col2:
        st.metric("Annual Spending", f"${spending_total:,.0f}", delta=f"{(spending_total/income_total*100):.0f}% of income")
    with col3:
        st.metric("Current Savings", f"${max(current_savings, 0):,.0f}")
    with col4:
        st.metric("Savings Rate", f"{savings_rate:.1f}%", delta="Good!" if savings_rate > 20 else "Needs improvement")
    
    st.markdown("---")
    
    # Main dashboard - two columns
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📊 Spending Breakdown")
        
        # Spending pie chart
        spending_data = user_transactions[user_transactions['amount'] < 0]
        if len(spending_data) > 0:
            spending_by_cat = spending_data.groupby('category')['amount'].sum().abs()
            
            # Create and display pie chart
            fig_pie = create_spending_pie_chart(spending_by_cat, persona_color)
            st.pyplot(fig_pie)
            plt.close(fig_pie)
        else:
            st.info("No spending data available")
        
        st.subheader("📈 Monthly Cash Flow")
        
        # Monthly cash flow chart
        user_transactions['month'] = pd.to_datetime(user_transactions['date']).dt.month
        monthly_income = user_transactions[user_transactions['amount'] > 0].groupby('month')['amount'].sum()
        monthly_spending = user_transactions[user_transactions['amount'] < 0].groupby('month')['amount'].sum().abs()
        
        # Ensure all months 1-12 are present
        for month in range(1, 13):
            if month not in monthly_income.index:
                monthly_income[month] = 0
            if month not in monthly_spending.index:
                monthly_spending[month] = 0
        
        monthly_income = monthly_income.sort_index()
        monthly_spending = monthly_spending.sort_index()
        
        fig_cashflow = create_monthly_cashflow_chart(monthly_income, monthly_spending, persona_color)
        st.pyplot(fig_cashflow)
        plt.close(fig_cashflow)
    
    with right_col:
        st.subheader("💡 Personalized Nudges")
        
        nudges = generate_nudges(user_features, persona, user_transactions)
        
        for nudge in nudges:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                <h4 style='margin: 0;'>{nudge['icon']} {nudge['title']}</h4>
                <p style='margin: 10px 0;'>{nudge['message']}</p>
                <button style='background-color: {persona_color}; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;'>
                    {nudge['action']}
                </button>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("🎯 Wealth Projection")
        
        # Monte Carlo simulation
        with st.spinner("Running Monte Carlo simulation..."):
            projection_10yr = monte_carlo_projection(
                current_savings=max(current_savings, 1000),
                monthly_savings=max(monthly_savings, 50),
                years=10
            )
            
            projection_20yr = monte_carlo_projection(
                current_savings=max(current_savings, 1000),
                monthly_savings=max(monthly_savings, 50),
                years=20
            )
        
        # Monte Carlo chart
        years = list(range(1, 21))
        medians = []
        p10s = []
        p90s = []
        
        progress_bar = st.progress(0)
        for i, y in enumerate(years):
            proj = monte_carlo_projection(
                current_savings=max(current_savings, 1000),
                monthly_savings=max(monthly_savings, 50),
                years=y,
                n_simulations=200  # Reduced for speed
            )
            medians.append(proj['median'])
            p10s.append(proj['p10'])
            p90s.append(proj['p90'])
            progress_bar.progress((i + 1) / len(years))
        
        progress_bar.empty()
        
        fig_mc = create_monte_carlo_chart(years, medians, p10s, p90s, persona_color)
        st.pyplot(fig_mc)
        plt.close(fig_mc)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("10-Year Median", f"${projection_10yr['median']:,.0f}")
        with col_b:
            st.metric("20-Year Median", f"${projection_20yr['median']:,.0f}")
    
    st.markdown("---")
    
    # Spending insights section
    st.subheader("🔍 Deep Dive: Spending Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        spending_data = user_transactions[user_transactions['amount'] < 0]
        if len(spending_data) > 0:
            top_merchant = spending_data.groupby('merchant')['amount'].sum().abs().nlargest(1)
            if len(top_merchant) > 0:
                st.info(f"🏪 **Top Merchant:** {top_merchant.index[0]}\n\n${top_merchant.values[0]:,.0f} total")
        else:
            st.info("🏪 No merchant data")
    
    with insight_col2:
        avg_transaction = user_transactions[user_transactions['amount'] < 0]['amount'].abs().mean()
        st.info(f"💳 **Avg Transaction:**\n\n${avg_transaction:.2f}")
    
    with insight_col3:
        spending_data = user_transactions[user_transactions['amount'] < 0]
        if len(spending_data) > 0:
            frequent_category = spending_data.groupby('category').size().nlargest(1)
            if len(frequent_category) > 0:
                st.info(f"🔄 **Most Frequent:**\n\n{frequent_category.index[0]}\n({frequent_category.values[0]} transactions)")
        else:
            st.info("🔄 No category data")
    
    # Comparison with peers
    st.subheader("📊 How You Compare to Peers")
    
    compare_col1, compare_col2, compare_col3 = st.columns(3)
    
    avg_savings_rate = features_df['savings_rate'].mean()
    avg_spending = features_df['total_spending'].mean()
    
    with compare_col1:
        savings_diff = savings_rate - (avg_savings_rate * 100)
        st.metric("Savings Rate vs Avg", f"{savings_rate:.1f}%", 
                  delta=f"{savings_diff:+.1f}% vs peer avg")
    
    with compare_col2:
        spending_diff = spending_total - avg_spending
        st.metric("Total Spending vs Avg", f"${spending_total:,.0f}",
                  delta=f"${spending_diff:+,.0f} vs peer avg")
    
    with compare_col3:
        persona_counts = features_df['persona'].value_counts()
        pct_same_persona = (persona_counts.get(persona, 0) / len(features_df)) * 100
        st.metric(f"Users with {persona} Persona", f"{persona_counts.get(persona, 0)} users",
                  delta=f"{pct_same_persona:.1f}% of total")
    
    # Ethics and privacy notice
    with st.expander("🔒 Ethics, Privacy, and Responsible AI (LO4)"):
        st.markdown("""
        ### Our Commitment to Responsible AI
        
        **Privacy by Design**
        - All data in this demo is synthetically generated
        - No real user data is stored or transmitted
        - In production, we implement end-to-end encryption and user consent
        
        **Transparency**
        - You always know why a nudge was shown to you
        - All recommendations come with clear explanations
        - No dark patterns or manipulative designs
        
        **Fairness & Bias Prevention**
        - Persona models are audited for demographic bias
        - Credit-related nudges are regulated and compliant
        - Appeals process for disputed recommendations
        
        **Financial Wellness Focus**
        - Nudges prioritize user benefit over bank profit
        - No high-risk product recommendations without warnings
        - Educational content always available
        """)
    
    # Download buttons
    st.markdown("---")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            label="📥 Download User Data (CSV)",
            data=users_df.to_csv(index=False),
            file_name="pulsepersona_users.csv",
            mime="text/csv"
        )
    with col_d2:
        st.download_button(
            label="📥 Download Transactions (CSV)",
            data=transactions_df.to_csv(index=False),
            file_name="pulsepersona_transactions.csv",
            mime="text/csv"
        )
    with col_d3:
        st.download_button(
            label="📥 Download Features (CSV)",
            data=features_df.to_csv(index=False),
            file_name="pulsepersona_features.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
