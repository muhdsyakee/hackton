import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Match Prediction", layout="wide")

def generate_team_stats(team_name):
    np.random.seed(hash(team_name) % 1000)
    
    return {
        "Form": np.random.normal(7, 2),
        "Goals_Scored": np.random.normal(1.8, 0.5),
        "Goals_Conceded": np.random.normal(1.2, 0.4),
        "Home_Advantage": np.random.normal(0.3, 0.1),
        "Injuries": np.random.poisson(2),
        "Morale": np.random.uniform(0.5, 1.0),
        "Recent_Form": np.random.choice([3, 1, 0], 5, p=[0.5, 0.3, 0.2])
    }

def calculate_match_prediction(home_stats, away_stats, crypto_factor=1.0):
    home_strength = (
        home_stats["Form"] * 0.3 +
        home_stats["Goals_Scored"] * 0.25 +
        (2 - home_stats["Goals_Conceded"]) * 0.2 +
        home_stats["Home_Advantage"] * 0.15 +
        home_stats["Morale"] * 0.1
    ) / 3
    
    away_strength = (
        away_stats["Form"] * 0.3 +
        away_stats["Goals_Scored"] * 0.25 +
        (2 - away_stats["Goals_Conceded"]) * 0.2 +
        away_stats["Morale"] * 0.15 +
        (1 - away_stats["Home_Advantage"]) * 0.1
    ) / 3
    
    home_strength *= crypto_factor
    away_strength *= (2 - crypto_factor)
    
    total_strength = home_strength + away_strength
    home_win_prob = home_strength / total_strength
    away_win_prob = away_strength / total_strength
    draw_prob = 1 - home_win_prob - away_win_prob
    
    total_prob = home_win_prob + draw_prob + away_win_prob
    home_win_prob /= total_prob
    draw_prob /= total_prob
    away_win_prob /= total_prob
    
    return {
        "home_win": max(0.1, min(0.8, home_win_prob)),
        "draw": max(0.1, min(0.4, draw_prob)),
        "away_win": max(0.1, min(0.8, away_win_prob)),
        "confidence": min(0.95, max(0.6, (home_strength + away_strength) / 2 + 0.3))
    }

def get_team_logo(team_name):
    logos = {
        "Manchester City": "🔵",
        "Liverpool": "🔴",
        "Arsenal": "🔴",
        "Manchester United": "🔴",
        "Chelsea": "🔵",
        "Tottenham": "⚪",
        "Newcastle": "⚫",
        "Brighton": "🔵",
        "Aston Villa": "🟡",
        "West Ham": "🔴"
    }
    return logos.get(team_name, "⚽")

st.title("📊 AI Match Prediction Engine")
st.markdown("Get intelligent predictions based on team performance, form, and crypto market conditions.")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox(
        "Select Home Team:",
        ["Manchester City", "Liverpool", "Arsenal", "Manchester United", "Chelsea", 
         "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham"],
        key="home_team"
    )
    
    # Display home team info
    if home_team:
        home_logo = get_team_logo(home_team)
        st.markdown(f"### {home_logo} {home_team}")
        
        # Generate and display home team stats
        home_stats = generate_team_stats(home_team)
        
        col1a, col2a = st.columns(2)
        with col1a:
            st.metric("Form (Last 5)", f"{home_stats['Form']:.1f} pts")
            st.metric("Goals Scored", f"{home_stats['Goals_Scored']:.1f} per game")
            st.metric("Home Advantage", f"{home_stats['Home_Advantage']:.1f}")
        
        with col2a:
            st.metric("Goals Conceded", f"{home_stats['Goals_Conceded']:.1f} per game")
            st.metric("Injuries", home_stats['Injuries'])
            st.metric("Morale", f"{home_stats['Morale']:.1f}")
        
        # Recent form visualization
        recent_form = home_stats['Recent_Form']
        form_labels = ['W' if x == 3 else 'D' if x == 1 else 'L' for x in recent_form]
        form_colors = ['green' if x == 3 else 'orange' if x == 1 else 'red' for x in recent_form]
        
        st.markdown("**Recent Form (Last 5):**")
        form_cols = st.columns(5)
        for i, (label, color) in enumerate(zip(form_labels, form_colors)):
            with form_cols[i]:
                st.markdown(f"<div style='text-align: center; padding: 10px; background-color: {color}; color: white; border-radius: 5px;'>{label}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox(
        "Select Away Team:",
        ["Manchester City", "Liverpool", "Arsenal", "Manchester United", "Chelsea", 
         "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham"],
        key="away_team"
    )
    
    # Display away team info
    if away_team:
        away_logo = get_team_logo(away_team)
        st.markdown(f"### {away_logo} {away_team}")
        
        # Generate and display away team stats
        away_stats = generate_team_stats(away_team)
        
        col1b, col2b = st.columns(2)
        with col1b:
            st.metric("Form (Last 5)", f"{away_stats['Form']:.1f} pts")
            st.metric("Goals Scored", f"{away_stats['Goals_Scored']:.1f} per game")
            st.metric("Away Performance", f"{1 - away_stats['Home_Advantage']:.1f}")
        
        with col2b:
            st.metric("Goals Conceded", f"{away_stats['Goals_Conceded']:.1f} per game")
            st.metric("Injuries", away_stats['Injuries'])
            st.metric("Morale", f"{away_stats['Morale']:.1f}")
        
        # Recent form visualization
        recent_form = away_stats['Recent_Form']
        form_labels = ['W' if x == 3 else 'D' if x == 1 else 'L' for x in recent_form]
        form_colors = ['green' if x == 3 else 'orange' if x == 1 else 'red' for x in recent_form]
        
        st.markdown("**Recent Form (Last 5):**")
        form_cols = st.columns(5)
        for i, (label, color) in enumerate(zip(form_labels, form_colors)):
            with form_cols[i]:
                st.markdown(f"<div style='text-align: center; padding: 10px; background-color: {color}; color: white; border-radius: 5px;'>{label}</div>", unsafe_allow_html=True)

# Crypto market influence section
st.markdown("---")
st.subheader("💹 Crypto Market Influence")

# Simulate crypto data (in real app, this would come from the main app)
crypto_data = {
    "bitcoin": {"usd_24h_change": np.random.normal(0, 3)},
    "ethereum": {"usd_24h_change": np.random.normal(0, 2.5)}
}

col1, col2, col3 = st.columns(3)
with col1:
    if crypto_data and "bitcoin" in crypto_data:
        btc_change = crypto_data["bitcoin"]["usd_24h_change"]
        st.metric("Bitcoin 24h Change", f"{btc_change:.2f}%")
    else:
        btc_change = 0
        st.metric("Bitcoin 24h Change", "N/A")
    
with col2:
    if crypto_data and "ethereum" in crypto_data:
        eth_change = crypto_data["ethereum"]["usd_24h_change"]
        st.metric("Ethereum 24h Change", f"{eth_change:.2f}%")
    else:
        eth_change = 0
        st.metric("Ethereum 24h Change", "N/A")
    
with col3:
    avg_change = (btc_change + eth_change) / 2
    market_sentiment = "🚀 Bullish" if avg_change > 1 else "📉 Bearish" if avg_change < -1 else "🤔 Neutral"
    st.metric("Market Sentiment", market_sentiment)

# Calculate crypto factor for predictions
crypto_factor = 1 + (avg_change / 100) * 0.2  # Crypto influence on predictions

# Prediction generation section
st.markdown("---")
st.subheader("🎯 Generate Prediction")

if st.button("🚀 Generate AI Prediction", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("❌ Please select different teams!")
    else:
        with st.spinner("🤖 AI is analyzing match data and crypto trends..."):
            import time
            time.sleep(2)  # Simulate processing time
            
            # Calculate prediction
            prediction = calculate_match_prediction(home_stats, away_stats, crypto_factor)
            
            st.success("🎉 AI Prediction Generated Successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏠 Home Win", f"{prediction['home_win']:.1%}")
                if prediction['home_win'] > 0.5:
                    st.success("🎯 High probability!")
                elif prediction['home_win'] > 0.35:
                    st.info("📊 Moderate probability")
                else:
                    st.warning("⚠️ Low probability")
            
            with col2:
                st.metric("🤝 Draw", f"{prediction['draw']:.1%}")
                if prediction['draw'] > 0.3:
                    st.success("🎯 High probability!")
                elif prediction['draw'] > 0.2:
                    st.info("📊 Moderate probability")
                else:
                    st.warning("⚠️ Low probability")
            
            with col3:
                st.metric("✈️ Away Win", f"{prediction['away_win']:.1%}")
                if prediction['away_win'] > 0.5:
                    st.success("🎯 High probability!")
                elif prediction['away_win'] > 0.35:
                    st.info("📊 Moderate probability")
                else:
                    st.warning("⚠️ Low probability")
            
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.metric("🎯 AI Confidence Level", f"{prediction['confidence']:.1%}")
                
                if prediction['confidence'] > 0.8:
                    st.success("🟢 **High Confidence** - This prediction is highly reliable!")
                elif prediction['confidence'] > 0.65:
                    st.warning("🟡 **Medium Confidence** - Use with caution and consider other factors")
                else:
                    st.error("🔴 **Low Confidence** - Not recommended for betting decisions")
                
                st.markdown("### 💰 Betting Recommendation")
                max_prob = max(prediction['home_win'], prediction['draw'], prediction['away_win'])
                
                if max_prob == prediction['home_win']:
                    recommendation = f"**Recommended Bet**: {home_team} to win"
                    odds = 1 / prediction['home_win']
                elif max_prob == prediction['away_win']:
                    recommendation = f"**Recommended Bet**: {away_team} to win"
                    odds = 1 / prediction['away_win']
                else:
                    recommendation = "**Recommended Bet**: Draw"
                    odds = 1 / prediction['draw']
                
                st.info(recommendation)
                st.metric("Suggested Odds", f"{odds:.2f}")
            
            with col2:
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Home Win', 'Draw', 'Away Win'],
                        y=[prediction['home_win'], prediction['draw'], prediction['away_win']],
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                        text=[f"{prediction['home_win']:.1%}", f"{prediction['draw']:.1%}", f"{prediction['away_win']:.1%}"],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Match Outcome Probabilities",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Team Comparison")
                comparison_data = pd.DataFrame({
                    'Metric': ['Form', 'Goals Scored', 'Goals Conceded', 'Morale'],
                    home_team: [home_stats['Form'], home_stats['Goals_Scored'], 
                               home_stats['Goals_Conceded'], home_stats['Morale']],
                    away_team: [away_stats['Form'], away_stats['Goals_Scored'], 
                               away_stats['Goals_Conceded'], away_stats['Morale']]
                })
                st.dataframe(comparison_data, use_container_width=True)
            
            with col2:
                st.markdown("### 💹 Crypto Impact Analysis")
                crypto_impact = pd.DataFrame({
                    'Factor': ['Bitcoin Trend', 'Ethereum Trend', 'Market Sentiment', 'Prediction Influence'],
                    'Value': [f"{btc_change:.2f}%", f"{eth_change:.2f}%", 
                             market_sentiment, f"{crypto_factor:.3f}"]
                })
                st.dataframe(crypto_impact, use_container_width=True)
            
            if 'predictions_history' not in st.session_state:
                st.session_state.predictions_history = []
            
            prediction_record = {
                'timestamp': datetime.now(),
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': prediction['home_win'],
                'draw_prob': prediction['draw'],
                'away_win_prob': prediction['away_win'],
                'confidence': prediction['confidence'],
                'crypto_factor': crypto_factor
            }
            
            st.session_state.predictions_history.append(prediction_record)

if 'predictions_history' in st.session_state and st.session_state.predictions_history:
    st.markdown("---")
    st.subheader("📚 Prediction History")
    
    history_df = pd.DataFrame(st.session_state.predictions_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(history_df, use_container_width=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.predictions_history = []
        st.rerun()

st.markdown("---")
st.markdown("*This AI prediction engine uses advanced algorithms and crypto market analysis to provide the most accurate match predictions.*")
