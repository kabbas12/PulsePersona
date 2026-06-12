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
import plotly.express as px
import plotly.graph_objects as go

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
    persona_list = list(PERSONA_PATTERNS.keys())
    
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
    """
    Return color for each persona
    """
    colors = {
        'Saver': '#2E86AB',  # Blue
        'Spender': '#F18F01',  # Orange
        'Investor': '#2D6A4F',  # Green
        'Debtor': '#D62828',  # Red
        'Balanced': '#6B4E71'  # Purple
    }
    return colors.get(persona, '#4A90E2')


def get_persona_icon(persona):
    """
    Return icon for each persona
    """
    icons = {
        'Saver': '🐷',
        'Spender': '💳',
        'Investor': '📈',
        'Debtor': '⚠️',
        'Balanced': '⚖️'
    }
    return icons.get(persona, '💜')


def get_persona_description(persona):
    """
    Return description for each persona
    """
    descriptions = {
        'Saver': "You're a natural saver who prioritizes financial security. You live below your means and consistently put money aside for the future.",
        'Spender': "You enjoy life's pleasures and aren't afraid to spend on experiences. The key is balancing enjoyment with future goals.",
        'Investor': "You understand that money should work for you. You're focused on building long-term wealth through strategic investments.",
        'Debtor': "You're currently carrying debt that's impacting your financial flexibility. The good news is there are clear paths to freedom.",
        'Balanced': "You've found a healthy equilibrium between spending today and saving for tomorrow. Keep optimizing!"
    }
    return descriptions.get(persona, "A unique financial personality with room to grow.")


# ============================================
# MAIN APP
# ============================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/4A90E2/white?text=PulsePersona", use_column_width=True)
        st.title("💜 PulsePersona")
        st.markdown("---")
        
        # Data generation
        st.subheader("📊 Data Settings")
        num_users = st.slider("Number of Users", 50, 500, 100, 50)
        transactions_per_user = st.slider("Transactions per User", 50, 200, 100, 50)
        
        if st.button("🔄 Generate / Refresh Data"):
            with st.spinner("Generating synthetic data..."):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        st.subheader("📚 Course Alignment")
        st.caption("LO1: FinTech & AI Role")
        st.caption("LO2: ML Techniques (Clustering)")
        st.caption("LO3: Strategic Analysis")
        st.caption("LO4: Ethics & Privacy")
        st.caption("LO5: Communication")
        st.caption("LO6: Team Collaboration")
        
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
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_ids = sorted(users_df['user_id'].unique())
        selected_user = st.selectbox("Select a User", user_ids, format_func=lambda x: f"User {x}")
    
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
    monthly_savings = current_savings / 12 if current_savings > 0 else 0
    
    # Persona header
    st.markdown(f"""
    <div style='background-color: {persona_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {persona_color};'>
        <h2>{persona_icon} {persona} Persona</h2>
        <p style='font-size: 16px;'>{get_persona_description(persona)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Income", f"${income_total:,.0f}")
    with col2:
        st.metric("Annual Spending", f"${spending_total:,.0f}", delta=f"{(spending_total/income_total*100):.0f}% of income")
    with col3:
        st.metric("Current Savings", f"${current_savings:,.0f}")
    with col4:
        savings_rate = (current_savings / income_total * 100) if income_total > 0 else 0
        st.metric("Savings Rate", f"{savings_rate:.1f}%", delta="Good!" if savings_rate > 20 else "Below target")
    
    st.markdown("---")
    
    # Main dashboard - two columns
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📊 Spending Breakdown")
        
        # Spending pie chart
        spending_data = user_transactions[user_transactions['amount'] < 0]
        if len(spending_data) > 0:
            spending_by_cat = spending_data.groupby('category')['amount'].sum().abs()
            
            fig = px.pie(
                values=spending_by_cat.values,
                names=spending_by_cat.index,
                title="Your Spending by Category",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No spending data available")
        
        st.subheader("📈 Monthly Cash Flow")
        
        # Monthly cash flow chart
        user_transactions['month'] = pd.to_datetime(user_transactions['date']).dt.to_period('M')
        monthly_income = user_transactions[user_transactions['amount'] > 0].groupby('month')['amount'].sum()
        monthly_spending = user_transactions[user_transactions['amount'] < 0].groupby('month')['amount'].sum().abs()
        
        cashflow_df = pd.DataFrame({
            'Income': monthly_income,
            'Spending': monthly_spending
        }).fillna(0)
        
        fig = px.bar(
            cashflow_df,
            title="Monthly Income vs Spending",
            labels={'value': 'Amount ($)', 'month': 'Month', 'variable': 'Type'},
            barmode='group',
            color_discrete_sequence=['#2E86AB', '#F18F01']
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        for y in years:
            proj = monte_carlo_projection(
                current_savings=max(current_savings, 1000),
                monthly_savings=max(monthly_savings, 50),
                years=y
            )
            medians.append(proj['median'])
            p10s.append(proj['p10'])
            p90s.append(proj['p90'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=medians,
            mode='lines', name='Median',
            line=dict(color=persona_color, width=3)
        ))
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=p90s + p10s[::-1],
            fill='toself',
            fillcolor=f'rgba({int(persona_color[1:3], 16)}, {int(persona_color[3:5], 16)}, {int(persona_color[5:7], 16)}, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='10th-90th Percentile'
        ))
        fig.update_layout(
            title='Wealth Projection (Monte Carlo Simulation)',
            xaxis_title='Years',
            yaxis_title='Portfolio Value ($)',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
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
        top_merchant = user_transactions[user_transactions['amount'] < 0].groupby('merchant')['amount'].sum().abs().nlargest(1)
        if len(top_merchant) > 0:
            st.info(f"🏪 **Top Merchant:** {top_merchant.index[0]}\n${top_merchant.values[0]:,.0f} total")
    
    with insight_col2:
        avg_transaction = user_transactions[user_transactions['amount'] < 0]['amount'].abs().mean()
        st.info(f"💳 **Avg Transaction:** ${avg_transaction:.2f}")
    
    with insight_col3:
        frequent_category = user_transactions[user_transactions['amount'] < 0].groupby('category').size().nlargest(1)
        if len(frequent_category) > 0:
            st.info(f"🔄 **Most Frequent:** {frequent_category.index[0]}\n{frequent_category.values[0]} transactions")
    
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
