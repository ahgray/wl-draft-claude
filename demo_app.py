#!/usr/bin/env python3
"""
NFL Draft Optimizer - Demo Version
Minimal version to show functionality without full dependency chain
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="NFL Draft Optimizer Demo",
    page_icon="🏈",
    layout="wide"
)

@st.cache_data
def load_massey_data():
    """Load Massey ratings data"""
    try:
        # Parse the CSV manually due to formatting issues
        data = []
        with open('massey-2025-export.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:33]:  # Skip header, get 32 teams
                parts = line.strip().split(',')
                if len(parts) >= 18:
                    team = parts[0].replace('﻿', '')  # Remove BOM
                    rating = float(parts[5]) if parts[5].replace('.', '').replace('-', '').isdigit() else 8.0
                    exp_wins = float(parts[17]) if len(parts) > 17 and parts[17].replace('.', '').replace('-', '').isdigit() else 8.0
                    exp_losses = float(parts[18]) if len(parts) > 18 and parts[18].replace('.', '').replace('-', '').isdigit() else 9.0
                    
                    data.append({
                        'team': team,
                        'rating': rating,
                        'expected_wins': exp_wins,
                        'expected_losses': exp_losses,
                        'playoff_prob': 0.5 if exp_wins > 9 else 0.2,
                        'draft_value': rating + (exp_wins - 8.5) * 0.5
                    })
        
        df = pd.DataFrame(data)
        return df.sort_values('rating', ascending=False)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def simulate_win_probability(rating_a, rating_b, home_advantage=2.65, k_factor=0.04):
    """Calculate win probability"""
    rating_diff = rating_a - rating_b + home_advantage
    return 1 / (1 + np.exp(-k_factor * rating_diff))

def get_snake_picks(position=6, n_drafters=8, n_rounds=4):
    """Calculate snake draft pick numbers"""
    picks = []
    for round_num in range(1, n_rounds + 1):
        if round_num % 2 == 1:  # Odd rounds
            pick = (round_num - 1) * n_drafters + position
        else:  # Even rounds
            pick = round_num * n_drafters - position + 1
        picks.append(pick)
    return picks

def main():
    st.title("🏈 NFL Draft Optimizer - Demo")
    
    # Load data
    team_data = load_massey_data()
    
    if team_data.empty:
        st.error("Could not load team data")
        return
    
    # Initialize session state
    if 'drafted_teams' not in st.session_state:
        st.session_state.drafted_teams = []
    if 'my_teams' not in st.session_state:
        st.session_state.my_teams = []
    if 'current_pick' not in st.session_state:
        st.session_state.current_pick = 1
    
    # Sidebar - Draft Controls
    with st.sidebar:
        st.header("Draft Controls")
        
        our_picks = get_snake_picks()
        st.write(f"**Our picks:** {our_picks}")
        
        current_round = (st.session_state.current_pick - 1) // 8 + 1
        st.metric("Current Pick", f"#{st.session_state.current_pick}")
        st.metric("Round", current_round)
        
        is_our_pick = st.session_state.current_pick in our_picks
        if is_our_pick:
            st.success("🎯 YOUR PICK!")
        else:
            next_our_pick = None
            for pick in our_picks:
                if pick > st.session_state.current_pick:
                    next_our_pick = pick
                    break
            if next_our_pick:
                picks_until = next_our_pick - st.session_state.current_pick
                st.info(f"Picks until your turn: {picks_until}")
        
        st.divider()
        
        # Record pick
        st.subheader("Record Pick")
        available_teams = [team for team in team_data['team'].tolist() 
                          if team not in st.session_state.drafted_teams]
        
        if available_teams:
            selected_team = st.selectbox("Team", available_teams)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Record Pick", type="primary"):
                    st.session_state.drafted_teams.append(selected_team)
                    if is_our_pick:
                        st.session_state.my_teams.append(selected_team)
                    st.session_state.current_pick += 1
                    st.rerun()
            
            with col2:
                if st.button("Undo"):
                    if st.session_state.drafted_teams:
                        last_team = st.session_state.drafted_teams.pop()
                        if last_team in st.session_state.my_teams:
                            st.session_state.my_teams.remove(last_team)
                        st.session_state.current_pick -= 1
                        st.rerun()
        
        st.divider()
        
        # My portfolio
        if st.session_state.my_teams:
            st.subheader("My Teams")
            my_data = team_data[team_data['team'].isin(st.session_state.my_teams)]
            total_wins = my_data['expected_wins'].sum()
            total_losses = len(st.session_state.my_teams) * 17 - total_wins
            
            st.metric("Expected Wins", f"{total_wins:.1f}")
            st.metric("Expected Losses", f"{total_losses:.1f}")
            
            for team in st.session_state.my_teams:
                st.write(f"• {team}")
    
    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "📋 Team Rankings", "📈 Analysis"])
    
    with tab1:
        st.header("Pick Recommendations")
        
        available_data = team_data[~team_data['team'].isin(st.session_state.drafted_teams)]
        
        if is_our_pick and not available_data.empty:
            st.subheader("🎯 Top Recommendations for Your Pick")
            
            # Simple recommendation logic
            if len(st.session_state.my_teams) == 0:
                # First pick - go for best available
                recommendations = available_data.nlargest(5, 'rating')
                strategy = "Best Available"
            elif len(st.session_state.my_teams) == 1:
                # Second pick - commit to strategy
                first_pick_wins = team_data[team_data['team'] == st.session_state.my_teams[0]]['expected_wins'].iloc[0]
                if first_pick_wins > 9:
                    recommendations = available_data.nlargest(5, 'expected_wins')
                    strategy = "Win Maximizer"
                elif first_pick_wins < 7:
                    recommendations = available_data.nsmallest(5, 'expected_wins')
                    strategy = "Loss Maximizer"
                else:
                    recommendations = available_data.nlargest(5, 'draft_value')
                    strategy = "Balanced"
            else:
                # Later picks - best available
                recommendations = available_data.nlargest(5, 'draft_value')
                strategy = "Best Available"
            
            st.info(f"**Strategy:** {strategy}")
            
            # Display top pick prominently
            if len(recommendations) > 0:
                top_pick = recommendations.iloc[0]
                st.success(f"### 🏆 TOP PICK: **{top_pick['team']}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rating", f"{top_pick['rating']:.2f}")
                with col2:
                    st.metric("Exp Wins", f"{top_pick['expected_wins']:.1f}")
                with col3:
                    st.metric("Exp Losses", f"{top_pick['expected_losses']:.1f}")
            
            # Show all recommendations
            st.dataframe(
                recommendations[['team', 'rating', 'expected_wins', 'expected_losses']].round(2),
                use_container_width=True,
                hide_index=True
            )
        
        elif not is_our_pick:
            st.info("Waiting for other drafters...")
            
            # Show what might be available at our next pick
            next_pick = None
            for pick in our_picks:
                if pick > st.session_state.current_pick:
                    next_pick = pick
                    break
            
            if next_pick:
                picks_between = next_pick - st.session_state.current_pick - 1
                st.subheader(f"Likely Available at Pick #{next_pick}")
                
                # Simple model - assume top teams get picked
                likely_gone = min(picks_between, len(available_data))
                likely_available = available_data.iloc[likely_gone:likely_gone+10]
                
                st.dataframe(
                    likely_available[['team', 'rating', 'expected_wins', 'expected_losses']].round(2),
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab2:
        st.header("Team Rankings")
        
        # Add availability column
        team_display = team_data.copy()
        team_display['available'] = ~team_display['team'].isin(st.session_state.drafted_teams)
        team_display['status'] = team_display['available'].map({True: '✅', False: '❌'})
        
        # Sort options
        sort_col = st.selectbox("Sort by", ['rating', 'expected_wins', 'expected_losses'])
        ascending = st.checkbox("Ascending", value=(sort_col == 'expected_losses'))
        
        display_data = team_display.sort_values(sort_col, ascending=ascending)
        
        st.dataframe(
            display_data[['status', 'team', 'rating', 'expected_wins', 'expected_losses', 'playoff_prob']],
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.header("Draft Analysis")
        
        if st.session_state.my_teams:
            # Portfolio analysis
            st.subheader("My Portfolio Analysis")
            
            my_data = team_data[team_data['team'].isin(st.session_state.my_teams)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Win Strategy Strength:**")
                win_total = my_data['expected_wins'].sum()
                win_rank = len(team_data) - my_data['expected_wins'].rank().sum() + len(my_data)
                st.metric("Total Expected Wins", f"{win_total:.1f}")
                st.metric("Avg Team Rank (Wins)", f"{win_rank/len(my_data):.1f}")
            
            with col2:
                st.write("**Loss Strategy Strength:**")
                loss_total = my_data['expected_losses'].sum()
                loss_rank = my_data['expected_losses'].rank().sum()
                st.metric("Total Expected Losses", f"{loss_total:.1f}")
                st.metric("Avg Team Rank (Losses)", f"{loss_rank/len(my_data):.1f}")
            
            # Show team details
            st.dataframe(
                my_data[['team', 'rating', 'expected_wins', 'expected_losses']],
                use_container_width=True,
                hide_index=True
            )
        
        else:
            st.info("Draft some teams to see portfolio analysis")
        
        # Draft progress
        st.subheader("Draft Progress")
        progress = len(st.session_state.drafted_teams) / 32
        st.progress(progress)
        st.write(f"Teams drafted: {len(st.session_state.drafted_teams)}/32")

if __name__ == "__main__":
    main()