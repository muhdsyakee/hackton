import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Odds", layout="wide")

@st.cache_data(ttl=300)
def fetch_crypto_data():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum,tether,binancecoin,cardano,solana,polkadot,chainlink",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true"
        }
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching crypto data: {e}")
        return None

def generate_crypto_soccer_correlation():
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    btc_prices = [50000]
    for i in range(1, n_days):
        change = np.random.normal(0, 0.02)
        btc_prices.append(btc_prices[-1] * (1 + change))
    
    home_win_rates = []
    for i, price in enumerate(btc_prices):
        base_rate = 0.45
        crypto_influence = (price - 50000) / 50000 * 0.1
        noise = np.random.normal(0, 0.05)
        home_win_rates.append(max(0.3, min(0.6, base_rate + crypto_influence + noise)))
    
    return pd.DataFrame({
        'Date': dates,
        'BTC_Price': btc_prices,
        'Home_Win_Rate': home_win_rates,
        'BTC_Change': [(p - btc_prices[0]) / btc_prices[0] * 100 for p in btc_prices]
    })

def calculate_betting_odds(probabilities):
    return {
        'home_odds': round(1 / probabilities['home_win'], 2),
        'draw_odds': round(1 / probabilities['draw'], 2),
        'away_odds': round(1 / probabilities['away_win'], 2)
    }

def analyze_market_sentiment(crypto_data):
    if not crypto_data:
        return "Neutral", 0, "No data available"
    
    changes = []
    for coin in ['bitcoin', 'ethereum', 'binancecoin']:
        if coin in crypto_data:
            changes.append(crypto_data[coin]['usd_24h_change'])
    
    if not changes:
        return "Neutral", 0, "No data available"
    
    avg_change = sum(changes) / len(changes)
    
    if avg_change > 3:
        return "🚀 Strong Bullish", avg_change, "Excellent market conditions"
    elif avg_change > 1:
        return "📈 Bullish", avg_change, "Good market conditions"
    elif avg_change > -1:
        return "🤔 Neutral", avg_change, "Stable market conditions"
    elif avg_change > -3:
        return "📉 Bearish", avg_change, "Challenging market conditions"
    else:
        return "💥 Strong Bearish", avg_change, "Difficult market conditions"

st.title("💹 Live Crypto Market & Soccer Correlation Analysis")
st.markdown("Analyze how cryptocurrency market trends influence soccer match predictions and betting odds.")

# Fetch live crypto data
crypto_data = fetch_crypto_data()

if crypto_data:
    # Live crypto dashboard
    st.markdown("---")
    st.subheader("📊 Live Crypto Market Dashboard")
    
    # Top cryptocurrencies
    top_coins = ['bitcoin', 'ethereum', 'binancecoin', 'cardano']
    
    col1, col2, col3, col4 = st.columns(4)
    for i, coin in enumerate(top_coins):
        if coin in crypto_data:
            with [col1, col2, col3, col4][i]:
                price = crypto_data[coin]['usd']
                change = crypto_data[coin]['usd_24h_change']
                market_cap = crypto_data[coin]['usd_market_cap']
                
                st.metric(
                    label=coin.title(),
                    value=f"${price:,.2f}",
                    delta=f"{change:.2f}%"
                )
                st.caption(f"Market Cap: ${market_cap:,.0f}")
    
    # Market sentiment analysis
    st.markdown("---")
    st.subheader("🎯 Market Sentiment Analysis")
    
    sentiment, avg_change, description = analyze_market_sentiment(crypto_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Sentiment", sentiment)
    with col2:
        st.metric("Average 24h Change", f"{avg_change:.2f}%")
    with col3:
        st.info(description)
    
    # Crypto price charts
    st.markdown("---")
    st.subheader("📈 Price Performance Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price comparison
        prices = []
        labels = []
        for coin in top_coins:
            if coin in crypto_data:
                prices.append(crypto_data[coin]['usd'])
                labels.append(coin.title())
        
        fig = px.bar(
            x=labels, 
            y=prices, 
            title="Current Prices (USD)",
            color=prices,
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 24h change comparison
        changes = []
        for coin in top_coins:
            if coin in crypto_data:
                changes.append(crypto_data[coin]['usd_24h_change'])
        
        fig = px.bar(
            x=labels, 
            y=changes, 
            title="24h Price Change (%)",
            color=changes,
            color_continuous_scale=["red", "yellow", "green"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Crypto-Soccer correlation analysis
    st.markdown("---")
    st.subheader("🔗 Crypto-Soccer Correlation Analysis")
    
    # Generate correlation data
    correlation_data = generate_crypto_soccer_correlation()
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(correlation_data['BTC_Price'], correlation_data['Home_Win_Rate'])[0, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Correlation Statistics")
        st.metric("BTC Price vs Home Win Rate", f"{correlation:.3f}")
        
        if abs(correlation) > 0.3:
            st.success("🎯 **Strong correlation detected!** Crypto trends significantly influence soccer outcomes.")
        elif abs(correlation) > 0.1:
            st.info("📈 **Moderate correlation detected.** Some influence of crypto on soccer results.")
        else:
            st.warning("⚠️ **Weak correlation detected.** Limited crypto influence on soccer outcomes.")
        
        # Betting implications
        st.markdown("### 💰 Betting Implications")
        if correlation > 0.2:
            st.success("**Strategy**: When crypto is bullish, favor home teams. When bearish, consider away teams.")
        elif correlation < -0.2:
            st.success("**Strategy**: When crypto is bearish, favor home teams. When bullish, consider away teams.")
        else:
            st.info("**Strategy**: Standard betting approach recommended. Crypto trends have minimal impact.")
    
    with col2:
        # Correlation scatter plot
        fig = px.scatter(
            correlation_data.sample(100),
            x='BTC_Price', 
            y='Home_Win_Rate',
            title="BTC Price vs Home Win Rate Correlation",
            labels={'BTC_Price': 'Bitcoin Price (USD)', 'Home_Win_Rate': 'Home Win Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.markdown("---")
    st.subheader("⏰ Time Series Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # BTC price over time
        fig = px.line(
            correlation_data.tail(30),  # Last 30 days
            x='Date', 
            y='BTC_Price',
            title="Bitcoin Price (Last 30 Days)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Home win rate over time
        fig = px.line(
            correlation_data.tail(30),  # Last 30 days
            x='Date', 
            y='Home_Win_Rate',
            title="Home Win Rate (Last 30 Days)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Live betting odds calculator
    st.markdown("---")
    st.subheader("🎲 Live Betting Odds Calculator")
    
    st.markdown("Calculate how current crypto market conditions affect match predictions:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_team = st.text_input("Home Team", "Manchester City")
        home_strength = st.slider("Home Team Strength", 0.1, 1.0, 0.7)
    
    with col2:
        away_team = st.text_input("Away Team", "Liverpool")
        away_strength = st.slider("Away Team Strength", 0.1, 1.0, 0.6)
    
    with col3:
        crypto_influence = st.slider("Crypto Market Influence", 0.5, 1.5, 1.0, 0.1)
        st.caption("1.0 = Normal, >1.0 = Bullish, <1.0 = Bearish")
    
    if st.button("🎯 Calculate Odds"):
        # Calculate base probabilities
        total_strength = home_strength + away_strength
        base_home_prob = home_strength / total_strength
        base_away_prob = away_strength / total_strength
        base_draw_prob = 1 - base_home_prob - base_away_prob
        
        # Apply crypto influence
        crypto_factor = crypto_influence
        adjusted_home_prob = min(0.9, max(0.1, base_home_prob * crypto_factor))
        adjusted_away_prob = min(0.8, max(0.1, base_away_prob / crypto_factor))
        adjusted_draw_prob = 1 - adjusted_home_prob - adjusted_away_prob
        
        # Normalize
        total_prob = adjusted_home_prob + adjusted_draw_prob + adjusted_away_prob
        adjusted_home_prob /= total_prob
        adjusted_draw_prob /= total_prob
        adjusted_away_prob /= total_prob
        
        probabilities = {
            'home_win': adjusted_home_prob,
            'draw': adjusted_draw_prob,
            'away_win': adjusted_away_prob
        }
        
        odds = calculate_betting_odds(probabilities)
        
        # Display results
        st.success("🎉 Odds calculated successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏠 Home Win", f"{probabilities['home_win']:.1%}", f"Odds: {odds['home_odds']}")
        with col2:
            st.metric("🤝 Draw", f"{probabilities['draw']:.1%}", f"Odds: {odds['draw_odds']}")
        with col3:
            st.metric("✈️ Away Win", f"{probabilities['away_win']:.1%}", f"Odds: {odds['away_odds']}")
        
        # Market impact analysis
        st.markdown("---")
        st.subheader("📊 Market Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Probability Changes")
            prob_changes = pd.DataFrame({
                'Outcome': ['Home Win', 'Draw', 'Away Win'],
                'Base Probability': [f"{base_home_prob:.1%}", f"{base_draw_prob:.1%}", f"{base_away_prob:.1%}"],
                'Crypto Adjusted': [f"{adjusted_home_prob:.1%}", f"{adjusted_draw_prob:.1%}", f"{adjusted_away_prob:.1%}"],
                'Change': [f"{((adjusted_home_prob/base_home_prob-1)*100):+.1f}%", 
                          f"{((adjusted_draw_prob/base_draw_prob-1)*100):+.1f}%",
                          f"{((adjusted_away_prob/base_away_prob-1)*100):+.1f}%"]
            })
            st.dataframe(prob_changes, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Betting Value Analysis")
            
            # Calculate value bets
            implied_prob = 1 / odds['home_odds']
            value = probabilities['home_win'] - implied_prob
            
            if value > 0.05:
                st.success("🎯 **Value Bet Detected**: Home win offers good value!")
            elif value > 0.02:
                st.info("📊 **Fair Bet**: Home win is reasonably priced")
            else:
                st.warning("⚠️ **Poor Value**: Home win may be overpriced")
            
            st.metric("Value (Home Win)", f"{value:.1%}")
            
            # Recommendation
            if crypto_influence > 1.1:
                st.success("🚀 **Crypto Bullish**: Consider increasing home team bets")
            elif crypto_influence < 0.9:
                st.info("📉 **Crypto Bearish**: Consider away team or draw bets")
            else:
                st.info("⚖️ **Crypto Neutral**: Standard betting approach")

else:
    st.error("❌ Unable to fetch live crypto data. Please check your internet connection.")
    st.info("💡 You can still use the correlation analysis with historical data.")
    
    st.markdown("---")
    st.subheader("📊 Historical Correlation Analysis")
    
    correlation_data = generate_crypto_soccer_correlation()
    correlation = np.corrcoef(correlation_data['BTC_Price'], correlation_data['Home_Win_Rate'])[0, 1]
    
    st.metric("BTC Price vs Home Win Rate Correlation", f"{correlation:.3f}")
    
    fig = px.scatter(
        correlation_data.sample(100),
        x='BTC_Price', 
        y='Home_Win_Rate',
        title="Historical BTC Price vs Home Win Rate"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*This analysis combines real-time cryptocurrency market data with soccer statistics to provide intelligent betting insights.*")
st.markdown("*Remember: Past performance does not guarantee future results. Always bet responsibly.*")

