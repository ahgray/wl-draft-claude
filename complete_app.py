#!/usr/bin/env python3
"""
NFL Draft Optimizer - Complete Working Version
Uses actual simulation engine with working dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append('src')

from data.data_fetcher import NFLDataFetcher
from simulation.win_probability import WinProbabilityModel

# Page config
st.set_page_config(
    page_title="NFL Draft Optimizer",
    page_icon="🏈",
    layout="wide"
)

@st.cache_data
def load_team_data():
    """Load and prepare team data"""
    fetcher = NFLDataFetcher()
    composite = fetcher.create_composite_rankings()
    
    # Add some calculated fields for better display
    composite['expected_losses'] = 17 - composite['vegas_wins']
    composite['playoff_prob'] = np.where(composite['vegas_wins'] > 9, 0.8, 0.3)
    composite['tank_prob'] = np.where(composite['vegas_wins'] < 7, 0.6, 0.1)
    composite['chaos_score'] = abs(composite['vegas_wins'] - 8.5) * 0.2 + np.random.uniform(0, 0.3, len(composite))
    
    return composite.sort_values('composite_rating', ascending=False)

def simulate_team_season(team_rating, n_games=17, opponent_avg=8.5):
    """Simple season simulation for a team"""
    model = WinProbabilityModel()
    wins = 0
    
    for game in range(n_games):
        # Simulate opponent (random around average)
        opp_rating = np.random.normal(opponent_avg, 1.0)
        is_home = np.random.choice([True, False])
        
        if model.simulate_game(team_rating, opp_rating, is_home):
            wins += 1
    
    return wins

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

def get_current_drafter(pick_number, n_drafters=8):
    """Calculate which drafter should pick based on snake draft order"""
    if pick_number < 1:
        return 1
    
    # Convert to 0-based indexing
    pick_index = pick_number - 1
    
    # Calculate round (0-based) and position in round (0-based)
    round_num = pick_index // n_drafters
    position_in_round = pick_index % n_drafters
    
    # Snake draft: odd rounds (0, 2, 4...) go forward 1-8
    # Even rounds (1, 3, 5...) go backward 8-1
    if round_num % 2 == 0:  # Odd round in 1-based counting
        drafter = position_in_round + 1
    else:  # Even round in 1-based counting
        drafter = n_drafters - position_in_round
    
    return drafter

def analyze_portfolio(teams, team_data):
    """Analyze a portfolio of teams"""
    if not teams:
        return {}
    
    portfolio_data = team_data[team_data['team_abbr'].isin(teams)]
    
    return {
        'total_rating': portfolio_data['composite_rating'].sum(),
        'total_expected_wins': portfolio_data['vegas_wins'].sum(),
        'total_expected_losses': portfolio_data['expected_losses'].sum(),
        'avg_rating': portfolio_data['composite_rating'].mean(),
        'consistency': 1 / (portfolio_data['composite_rating'].std() + 0.1),
        'win_upside': portfolio_data['vegas_wins'].max(),
        'loss_upside': portfolio_data['expected_losses'].max(),
        'chaos_potential': portfolio_data['chaos_score'].mean()
    }

def get_strategy_recommendation(my_teams, team_data):
    """Recommend strategy based on current portfolio"""
    if not my_teams:
        return "FLEXIBLE", "Take best available or commit to a strategy"
    
    portfolio_wins = team_data[team_data['team_abbr'].isin(my_teams)]['vegas_wins'].mean()
    
    if portfolio_wins > 9.5:
        return "WIN_MAXIMIZER", "Continue targeting high-win teams"
    elif portfolio_wins < 7.5:
        return "LOSS_MAXIMIZER", "Continue targeting high-loss teams"
    elif len(my_teams) == 1:
        return "DECISION_POINT", "Commit to wins or losses strategy"
    else:
        return "BALANCED", "Maximize expected value either direction"

def main():
    st.title("🏈 NFL Draft Optimizer - Live Draft Tool")
    st.markdown("**Position #6 Snake Draft** | Pick Order: 6, 11, 22, 27")
    
    # Load data
    with st.spinner("Loading team data and simulations..."):
        team_data = load_team_data()
    
    # Initialize session state
    if 'drafted_teams' not in st.session_state:
        st.session_state.drafted_teams = []
    if 'my_teams' not in st.session_state:
        st.session_state.my_teams = []
    if 'current_pick' not in st.session_state:
        st.session_state.current_pick = 1
    if 'drafter_teams' not in st.session_state:
        st.session_state.drafter_teams = {i: [] for i in range(1, 9)}
    
    # Sidebar - Draft Controls
    with st.sidebar:
        st.header("🎯 Draft Control Center")
        
        # Current state
        our_picks = get_snake_picks()
        current_round = ((st.session_state.current_pick - 1) // 8) + 1
        is_our_pick = st.session_state.current_pick in our_picks
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pick #", st.session_state.current_pick)
        with col2:
            st.metric("Round", current_round)
        
        if is_our_pick:
            st.success("🎯 **YOUR PICK!**")
        else:
            next_our_pick = next((p for p in our_picks if p > st.session_state.current_pick), None)
            if next_our_pick:
                picks_until = next_our_pick - st.session_state.current_pick
                st.info(f"⏱️ {picks_until} picks until your turn")
        
        st.markdown("**Our picks:** " + " • ".join(map(str, our_picks)))
        
        st.divider()
        
        # Record pick
        st.subheader("📝 Record Pick")
        available_teams = [t for t in team_data['team_abbr'].tolist() 
                          if t not in st.session_state.drafted_teams]
        
        if available_teams:
            selected_team = st.selectbox("Team", available_teams, 
                                       help="Select the team that was just drafted")
            
            # Auto-detect drafter using correct snake draft order
            current_drafter = get_current_drafter(st.session_state.current_pick)
            drafter_id = st.number_input("Drafter", 1, 8, value=current_drafter)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Record Pick", type="primary"):
                    st.session_state.drafted_teams.append(selected_team)
                    st.session_state.drafter_teams[drafter_id].append(selected_team)
                    if is_our_pick:
                        st.session_state.my_teams.append(selected_team)
                    st.session_state.current_pick += 1
                    st.rerun()
            
            with col2:
                if st.button("↶ Undo") and st.session_state.current_pick > 1:
                    if st.session_state.drafted_teams:
                        last_team = st.session_state.drafted_teams.pop()
                        # Find and remove from drafter
                        for did, teams in st.session_state.drafter_teams.items():
                            if last_team in teams:
                                teams.remove(last_team)
                                break
                        if last_team in st.session_state.my_teams:
                            st.session_state.my_teams.remove(last_team)
                        st.session_state.current_pick -= 1
                        st.rerun()
        
        st.divider()
        
        # My portfolio
        if st.session_state.my_teams:
            st.subheader("🏆 My Portfolio")
            portfolio = analyze_portfolio(st.session_state.my_teams, team_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Wins", f"{portfolio['total_expected_wins']:.1f}")
                st.metric("Avg Rating", f"{portfolio['avg_rating']:.2f}")
            with col2:
                st.metric("Expected Losses", f"{portfolio['total_expected_losses']:.1f}")
                st.metric("Consistency", f"{portfolio['consistency']:.2f}")
            
            for i, team in enumerate(st.session_state.my_teams, 1):
                team_info = team_data[team_data['team_abbr'] == team].iloc[0]
                st.write(f"{i}. **{team}** ({team_info['vegas_wins']:.1f}W)")
        else:
            st.info("No teams drafted yet")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Recommendations", "🏈 Team Rankings", "📈 Draft Board", "🧠 Strategy"])
    
    with tab1:
        st.header("📊 Pick Recommendations")
        
        available_data = team_data[~team_data['team_abbr'].isin(st.session_state.drafted_teams)]
        strategy, strategy_desc = get_strategy_recommendation(st.session_state.my_teams, team_data)
        
        if is_our_pick and not available_data.empty:
            # Get recommendations based on strategy
            if strategy == "WIN_MAXIMIZER":
                recommendations = available_data.nlargest(8, 'vegas_wins')
            elif strategy == "LOSS_MAXIMIZER":
                recommendations = available_data.nlargest(8, 'expected_losses')
            else:
                recommendations = available_data.nlargest(8, 'composite_rating')
            
            # Show strategy
            strategy_colors = {
                "WIN_MAXIMIZER": "🟢",
                "LOSS_MAXIMIZER": "🔴", 
                "DECISION_POINT": "🟡",
                "BALANCED": "🔵",
                "FLEXIBLE": "⚪"
            }
            
            st.info(f"{strategy_colors.get(strategy, '⚪')} **Strategy: {strategy}** - {strategy_desc}")
            
            # Top recommendation
            if len(recommendations) > 0:
                top_pick = recommendations.iloc[0]
                
                st.success(f"### 🎯 TOP PICK: **{top_pick['team_abbr']}**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rating", f"{top_pick['composite_rating']:.2f}")
                with col2:
                    st.metric("Expected Wins", f"{top_pick['vegas_wins']:.1f}")
                with col3:
                    st.metric("Expected Losses", f"{top_pick['expected_losses']:.1f}")
                with col4:
                    st.metric("Vegas O/U", f"{top_pick['vegas_wins']:.1f}")
            
            # All recommendations table
            st.subheader("🏈 All Recommendations")
            display_recs = recommendations[['team_abbr', 'composite_rating', 'vegas_wins', 
                                         'expected_losses', 'playoff_prob', 'tank_prob']].copy()
            display_recs.columns = ['Team', 'Rating', 'Exp Wins', 'Exp Losses', 'Playoff %', 'Tank %']
            
            st.dataframe(
                display_recs.round(2),
                use_container_width=True,
                hide_index=True
            )
        
        elif not is_our_pick:
            # Show what might be available
            st.info(f"⏳ Waiting for pick #{st.session_state.current_pick}")
            
            next_pick = next((p for p in our_picks if p > st.session_state.current_pick), None)
            if next_pick and not available_data.empty:
                picks_between = next_pick - st.session_state.current_pick - 1
                
                st.subheader(f"🔮 Likely Available at Pick #{next_pick}")
                st.caption(f"Assuming {picks_between} teams drafted before your turn")
                
                # Simple availability model
                likely_gone = min(picks_between, len(available_data))
                likely_available = available_data.iloc[likely_gone:likely_gone+10]
                
                st.dataframe(
                    likely_available[['team_abbr', 'composite_rating', 'vegas_wins', 'expected_losses']].round(2),
                    use_container_width=True,
                    hide_index=True
                )
        
        else:
            st.warning("⚠️ All teams have been drafted!")
    
    with tab2:
        st.header("🏈 Complete Team Rankings")
        
        # Add availability status
        team_display = team_data.copy()
        team_display['Status'] = team_display['team_abbr'].apply(
            lambda x: '❌ Drafted' if x in st.session_state.drafted_teams else '✅ Available'
        )
        
        # Sort controls
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_options = {
                'Composite Rating': 'composite_rating',
                'Expected Wins': 'vegas_wins', 
                'Expected Losses': 'expected_losses',
                'Playoff Probability': 'playoff_prob'
            }
            sort_by = st.selectbox("Sort by", list(sort_options.keys()))
        with col2:
            ascending = st.checkbox("Ascending", value=False)
        
        # Display data
        display_data = team_display.sort_values(sort_options[sort_by], ascending=ascending)
        
        columns_to_show = ['Status', 'team_abbr', 'composite_rating', 'vegas_wins', 
                          'expected_losses', 'playoff_prob', 'tank_prob']
        column_names = ['Status', 'Team', 'Rating', 'Exp Wins', 'Exp Losses', 'Playoff %', 'Tank %']
        
        display_df = display_data[columns_to_show].copy()
        display_df.columns = column_names
        
        st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("📈 Draft Board")
        
        # Draft progress
        progress = len(st.session_state.drafted_teams) / 32
        st.progress(progress, text=f"Draft Progress: {len(st.session_state.drafted_teams)}/32 teams")
        
        # Drafter analysis
        if len(st.session_state.drafted_teams) > 0:
            st.subheader("👥 Drafter Portfolios")
            
            drafter_analysis = []
            for drafter_id, teams in st.session_state.drafter_teams.items():
                if teams:
                    portfolio = analyze_portfolio(teams, team_data)
                    drafter_analysis.append({
                        'Drafter': f"{'YOU' if drafter_id == 6 else f'#{drafter_id}'}",
                        'Teams': len(teams),
                        'Expected Wins': portfolio['total_expected_wins'],
                        'Expected Losses': portfolio['total_expected_losses'],
                        'Avg Rating': portfolio['avg_rating'],
                        'Strategy': 'Wins' if portfolio['total_expected_wins'] > len(teams) * 8.5 else 'Losses'
                    })
            
            if drafter_analysis:
                analysis_df = pd.DataFrame(drafter_analysis)
                
                # Highlight our row
                def highlight_us(row):
                    return ['background-color: lightgreen' if row['Drafter'] == 'YOU' else '' for _ in row]
                
                styled_df = analysis_df.style.apply(highlight_us, axis=1)
                st.dataframe(styled_df.format({
                    'Expected Wins': '{:.1f}',
                    'Expected Losses': '{:.1f}', 
                    'Avg Rating': '{:.2f}'
                }), use_container_width=True, hide_index=True)
                
                # Visualization
                if len(analysis_df) > 1:
                    fig = px.scatter(analysis_df, 
                                   x='Expected Wins', 
                                   y='Expected Losses',
                                   text='Drafter',
                                   title="Drafter Positioning",
                                   color='Strategy')
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Recent picks
        if st.session_state.drafted_teams:
            st.subheader("📜 Recent Picks")
            recent_picks = st.session_state.drafted_teams[-10:]  # Last 10 picks
            
            pick_details = []
            for i, team in enumerate(recent_picks):
                pick_num = len(st.session_state.drafted_teams) - len(recent_picks) + i + 1
                team_info = team_data[team_data['team_abbr'] == team].iloc[0]
                pick_details.append({
                    'Pick': f"#{pick_num}",
                    'Team': team,
                    'Expected Wins': team_info['vegas_wins'],
                    'Rating': team_info['composite_rating']
                })
            
            st.dataframe(pd.DataFrame(pick_details).round(2), 
                        use_container_width=True, hide_index=True)
    
    with tab4:
        st.header("🧠 Strategy Analysis")
        
        # Overall strategy guidance
        st.subheader("📋 Strategy Guidance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Win Strategy Teams:**")
            win_targets = team_data.nlargest(8, 'vegas_wins')[['team_abbr', 'vegas_wins']]
            for _, team in win_targets.iterrows():
                status = "❌" if team['team_abbr'] in st.session_state.drafted_teams else "✅"
                st.write(f"{status} {team['team_abbr']} ({team['vegas_wins']:.1f}W)")
        
        with col2:
            st.markdown("**🎯 Loss Strategy Teams:**")
            loss_targets = team_data.nsmallest(8, 'vegas_wins')[['team_abbr', 'expected_losses']]
            for _, team in loss_targets.iterrows():
                status = "❌" if team['team_abbr'] in st.session_state.drafted_teams else "✅"
                st.write(f"{status} {team['team_abbr']} ({team['expected_losses']:.1f}L)")
        
        # Position-specific advice
        st.subheader("🎯 Position #6 Strategy")
        
        pick_advice = {
            6: "**Pick #6 (R1):** Take best available elite team. Lions, Rams likely targets if top 5 gone.",
            11: "**Pick #11 (R2):** Commit to strategy. If took elite team, continue wins. If took tank team, continue losses.",
            22: "**Pick #22 (R3):** Value pick opportunity. Look for teams that fell or block opponents.",
            27: "**Pick #27 (R4):** Complete your portfolio. Balance or double-down based on needs."
        }
        
        next_our_pick = next((p for p in our_picks if p >= st.session_state.current_pick), None)
        
        for pick, advice in pick_advice.items():
            if pick == next_our_pick:
                st.success(f"➡️ {advice}")
            else:
                st.info(advice)
        
        # Simulation results
        if st.session_state.my_teams:
            st.subheader("🔮 Portfolio Simulation")
            
            portfolio_analysis = analyze_portfolio(st.session_state.my_teams, team_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Win Strategy Rank", 
                         f"#{int(33 - portfolio_analysis['total_expected_wins'])}")
            with col2:
                st.metric("Loss Strategy Rank", 
                         f"#{int(portfolio_analysis['total_expected_losses'])}")
            with col3:
                st.metric("Chaos Potential", 
                         f"{portfolio_analysis['chaos_potential']:.2f}")

if __name__ == "__main__":
    main()