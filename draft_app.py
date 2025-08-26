"""
NFL Draft Optimizer - Streamlit UI
Real-time draft decision support tool
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from data.data_fetcher import NFLDataFetcher
from simulation.win_probability import WinProbabilityModel
from simulation.monte_carlo import MonteCarloSimulator, SimulationAnalyzer
from optimization.draft_strategy import DraftStrategyEngine, DraftState, Strategy, OpponentModeler
from optimization.draft_tracker import DraftTracker, DraftAnalyzer

# Page config
st.set_page_config(
    page_title="NFL Draft Optimizer",
    page_icon="🏈",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.tracker = DraftTracker(n_drafters=8, teams_per_drafter=4, my_position=6)
    st.session_state.strategy_engine = DraftStrategyEngine(draft_position=6)
    st.session_state.opponent_modeler = OpponentModeler()
    st.session_state.current_strategy = Strategy.BALANCED

@st.cache_data
def load_data():
    """Load and cache all data"""
    fetcher = NFLDataFetcher()
    data = fetcher.load_cached_data()
    return data

@st.cache_data
def run_simulations(rankings_df, schedule_df):
    """Run Monte Carlo simulations"""
    # Convert rankings to dict
    team_ratings = dict(zip(rankings_df['team_abbr'], rankings_df['composite_rating']))
    
    # Run simulations
    simulator = MonteCarloSimulator(n_simulations=50000)
    team_stats = simulator.run_full_simulation(
        schedule_df, 
        team_ratings,
        cache_path="data/cache/simulation_results.pkl"
    )
    
    return team_stats

def main():
    st.title("🏈 NFL Draft Optimizer - Position #6")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
        rankings = data['rankings']
        schedule = data['schedule']
    
    # Run simulations
    with st.spinner("Running simulations..."):
        team_stats = run_simulations(rankings, schedule)
    
    # Initialize tracker with teams
    if not st.session_state.initialized:
        st.session_state.tracker.initialize_teams(list(rankings['team_abbr']))
        st.session_state.initialized = True
    
    # Create analyzer
    analyzer = DraftAnalyzer(st.session_state.tracker, team_stats)
    
    # Sidebar - Draft Controls
    with st.sidebar:
        st.header("Draft Controls")
        
        # Current draft state
        draft_state = st.session_state.tracker.get_draft_state()
        
        st.metric("Current Pick", f"#{draft_state['current_pick'] + 1}")
        st.metric("Round", draft_state['current_round'])
        
        if draft_state['is_my_pick']:
            st.success("🎯 YOUR PICK!")
        else:
            st.info(f"Picks until your turn: {draft_state['picks_until_my_turn']}")
        
        st.divider()
        
        # Make a pick
        st.subheader("Record Pick")
        
        available_teams = sorted(draft_state['available_teams'])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_team = st.selectbox("Team", available_teams)
        with col2:
            drafter_id = st.number_input("Drafter", 1, 8, 
                                        value=draft_state['current_drafter'] or 1)
        
        if st.button("Record Pick", type="primary", use_container_width=True):
            try:
                pick_detail = st.session_state.tracker.make_pick(selected_team, drafter_id)
                st.success(f"Recorded: {selected_team} to Drafter {drafter_id}")
                
                # Update opponent model
                st.session_state.opponent_modeler.update_opponent_model(
                    drafter_id, selected_team, team_stats
                )
                
                st.rerun()
            except Exception as e:
                st.error(str(e))
        
        # Undo button
        if st.button("Undo Last Pick"):
            undone = st.session_state.tracker.undo_last_pick()
            if undone:
                st.info(f"Undone: {undone['team']}")
                st.rerun()
        
        st.divider()
        
        # Save/Load draft
        st.subheader("Draft Management")
        
        if st.button("Save Draft"):
            st.session_state.tracker.save_draft("draft_save.json")
            st.success("Draft saved!")
        
        if st.button("Load Draft"):
            try:
                st.session_state.tracker.load_draft("draft_save.json")
                st.success("Draft loaded!")
                st.rerun()
            except:
                st.error("No saved draft found")
    
    # Main area - tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Recommendations", "👥 Draft Board", "📈 Analysis", "🎯 Strategy", "📋 Team Rankings"]
    )
    
    with tab1:
        st.header("Pick Recommendations")
        
        if draft_state['is_my_pick']:
            # Get recommendations
            current_state = DraftState(
                round=draft_state['current_round'],
                pick_number=draft_state['current_pick'] + 1,
                available_teams=draft_state['available_teams'],
                my_teams=draft_state['my_teams'],
                opponent_teams=st.session_state.tracker.drafter_teams,
                current_strategy=st.session_state.current_strategy
            )
            
            recommendations = st.session_state.strategy_engine.get_recommendations(
                current_state, team_stats, n_recommendations=8
            )
            
            # Display top recommendation
            if len(recommendations) > 0:
                top_pick = recommendations.iloc[0]
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.success(f"### 🎯 Top Pick: **{top_pick['team']}**")
                with col2:
                    st.metric("Expected Wins", f"{top_pick['mean_wins']:.1f}")
                with col3:
                    st.metric("Value Score", f"{top_pick['value']:.1f}")
                
                # Show all recommendations
                st.subheader("All Recommendations")
                
                # Add team details
                recommendations = recommendations.merge(
                    team_stats[['team', 'p90_wins', 'p90_losses', 'std_wins']], 
                    on='team', 
                    how='left'
                )
                
                # Format for display
                display_cols = ['team', 'value', 'mean_wins', 'mean_losses', 
                               'p90_wins', 'p90_losses', 'strategy']
                
                st.dataframe(
                    recommendations[display_cols].round(2),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info(f"Waiting for pick #{draft_state['current_pick'] + 1}")
            
            # Show what's likely to be available
            st.subheader("Teams Likely Available at Your Next Pick")
            
            available_stats = team_stats[team_stats['team'].isin(draft_state['available_teams'])]
            
            # Simple probability model
            picks_until = draft_state['picks_until_my_turn']
            if picks_until > 0:
                available_stats['reach_prob'] = available_stats['mean_wins'].rank(pct=True)
                available_stats['still_available_prob'] = available_stats['reach_prob'] ** (picks_until / 10)
                
                likely_available = available_stats[available_stats['still_available_prob'] > 0.3].nlargest(
                    10, 'mean_wins'
                )[['team', 'mean_wins', 'mean_losses', 'still_available_prob']]
                
                st.dataframe(likely_available.round(2), use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("Draft Board")
        
        # Current portfolios
        drafter_summary = analyzer.analyze_drafter_strategies()
        
        if not drafter_summary.empty:
            # Highlight our row
            def highlight_me(row):
                if row['drafter_id'] == 6:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            styled_summary = drafter_summary.style.apply(highlight_me, axis=1)
            st.dataframe(styled_summary, use_container_width=True, hide_index=True)
        
        # Pick history
        st.subheader("Pick History")
        if st.session_state.tracker.pick_history:
            history_df = pd.DataFrame(st.session_state.tracker.pick_history)
            history_df = history_df[['pick_number', 'round', 'drafter_id', 'team']]
            st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("Draft Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio strength comparison
            st.subheader("Portfolio Strength")
            
            if not drafter_summary.empty:
                fig = px.scatter(
                    drafter_summary,
                    x='total_wins',
                    y='total_losses',
                    text='drafter_id',
                    title="Win/Loss Position",
                    labels={'total_wins': 'Expected Wins', 'total_losses': 'Expected Losses'}
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strategy distribution
            st.subheader("Apparent Strategies")
            
            if not drafter_summary.empty:
                strategy_counts = drafter_summary['strategy'].value_counts()
                fig = px.pie(
                    values=strategy_counts.values,
                    names=strategy_counts.index,
                    title="Strategy Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Value picks still available
        st.subheader("Value Picks Available")
        value_picks = analyzer.identify_value_picks()
        if not value_picks.empty:
            st.dataframe(value_picks.round(2), use_container_width=True, hide_index=True)
    
    with tab4:
        st.header("Strategy Control")
        
        # Current strategy
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Current Strategy")
            
            strategy_options = [s.value for s in Strategy]
            selected_strategy = st.selectbox(
                "Override Strategy",
                strategy_options,
                index=strategy_options.index(st.session_state.current_strategy.value)
            )
            
            if st.button("Update Strategy"):
                st.session_state.current_strategy = Strategy(selected_strategy)
                st.success(f"Strategy updated to: {selected_strategy}")
        
        with col2:
            st.subheader("My Portfolio Analysis")
            
            if draft_state['my_teams']:
                portfolio_strength = analyzer.calculate_portfolio_strength(draft_state['my_teams'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Wins", f"{portfolio_strength.get('total_expected_wins', 0):.1f}")
                with col2:
                    st.metric("Expected Losses", f"{portfolio_strength.get('total_expected_losses', 0):.1f}")
                with col3:
                    st.metric("Consistency", f"{portfolio_strength.get('consistency', 0):.2f}")
                
                st.write("**My Teams:**", ", ".join(draft_state['my_teams']))
            else:
                st.info("No teams drafted yet")
        
        # Opponent threats
        st.subheader("Opponent Threat Analysis")
        
        threat_data = []
        for drafter_id, profile in st.session_state.opponent_modeler.opponent_profiles.items():
            if drafter_id != 6:  # Not us
                threat = st.session_state.opponent_modeler.calculate_threat_level(drafter_id)
                threat_data.append({
                    'Drafter': drafter_id,
                    'Teams': ", ".join(profile['teams']),
                    'Strategy': profile.get('strategy', 'UNKNOWN'),
                    'Threat Level': threat
                })
        
        if threat_data:
            threat_df = pd.DataFrame(threat_data)
            threat_df = threat_df.sort_values('Threat Level', ascending=False)
            st.dataframe(threat_df, use_container_width=True, hide_index=True)
    
    with tab5:
        st.header("Team Rankings")
        
        # Merge team stats with rankings
        display_stats = team_stats[['team', 'mean_wins', 'mean_losses', 'std_wins', 
                                    'p90_wins', 'p90_losses', 'playoff_prob', 'top_pick_prob']].copy()
        
        # Add availability
        display_stats['available'] = display_stats['team'].isin(draft_state['available_teams'])
        
        # Sort options
        sort_by = st.selectbox("Sort by", ['mean_wins', 'mean_losses', 'std_wins', 'playoff_prob'])
        ascending = st.checkbox("Ascending", value=False)
        
        display_stats = display_stats.sort_values(sort_by, ascending=ascending)
        
        # Color code by availability
        def color_available(row):
            if not row['available']:
                return ['background-color: #FFB6C1'] * len(row)  # Light red for taken
            return [''] * len(row)
        
        styled_stats = display_stats.style.apply(color_available, axis=1).format({
            'mean_wins': '{:.1f}',
            'mean_losses': '{:.1f}',
            'std_wins': '{:.2f}',
            'p90_wins': '{:.1f}',
            'p90_losses': '{:.1f}',
            'playoff_prob': '{:.2%}',
            'top_pick_prob': '{:.2%}'
        })
        
        st.dataframe(styled_stats, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()