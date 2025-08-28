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
from analysis.post_draft import PostDraftAnalyzer

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

def get_snake_picks(position=None, n_drafters=8, n_rounds=4):
    """Calculate snake draft pick numbers for any position"""
    if position is None:
        position = st.session_state.get('draft_position', 1)
    
    picks = []
    for round_num in range(1, n_rounds + 1):
        if round_num % 2 == 1:  # Odd rounds
            pick = (round_num - 1) * n_drafters + position
        else:  # Even rounds
            pick = round_num * n_drafters - position + 1
        picks.append(pick)
    return picks

def generate_pick_advice(draft_position, n_drafters=8):
    """Generate position-specific draft advice"""
    picks = get_snake_picks(draft_position, n_drafters, 4)
    advice = {}
    
    # Position-specific strategy hints
    if draft_position <= 2:
        position_strategy = "elite teams available"
    elif draft_position <= 4:
        position_strategy = "strong tier 1 options"
    elif draft_position <= 6:
        position_strategy = "maximum flexibility"
    else:
        position_strategy = "value plays and snake advantage"
    
    for i, pick in enumerate(picks):
        round_num = i + 1
        if round_num == 1:
            advice[pick] = f"**Pick #{pick} (R1):** Best available - {position_strategy}"
        elif round_num == 2:
            advice[pick] = f"**Pick #{pick} (R2):** Commit to strategy based on R1 pick"
        elif round_num == 3:
            advice[pick] = f"**Pick #{pick} (R3):** Value pick or block opponents"
        else:
            advice[pick] = f"**Pick #{pick} (R4):** Complete portfolio or double-down"
    
    return advice

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
    
    # Initialize session state
    if 'draft_position' not in st.session_state:
        st.session_state.draft_position = None
    if 'drafted_teams' not in st.session_state:
        st.session_state.drafted_teams = []
    if 'my_teams' not in st.session_state:
        st.session_state.my_teams = []
    if 'current_pick' not in st.session_state:
        st.session_state.current_pick = 1
    if 'drafter_teams' not in st.session_state:
        st.session_state.drafter_teams = {i: [] for i in range(1, 9)}
    
    # Display current draft position info if set
    if st.session_state.draft_position:
        picks = get_snake_picks(st.session_state.draft_position)
        st.markdown(f"**Position #{st.session_state.draft_position} Snake Draft** | Pick Order: {', '.join(map(str, picks))}")
    else:
        st.warning("⚠️ Please select your draft position in the sidebar to begin")
    
    # Load data
    with st.spinner("Loading team data and simulations..."):
        team_data = load_team_data()
    
    # Sidebar - Draft Controls
    with st.sidebar:
        st.header("🎯 Draft Control Center")
        
        # Draft position selector
        st.subheader("📍 Your Draft Position")
        draft_pos = st.selectbox(
            "Select your draft position:",
            options=[None] + list(range(1, 9)),
            index=0 if st.session_state.draft_position is None else st.session_state.draft_position,
            format_func=lambda x: "Select position..." if x is None else f"Position {x}",
            help="Your draft position determines when you pick in each round"
        )
        
        if draft_pos != st.session_state.draft_position:
            st.session_state.draft_position = draft_pos
            st.rerun()
        
        if st.session_state.draft_position is None:
            st.error("Please select your draft position above")
            st.stop()
        
        st.divider()
        
        # Current state
        our_picks = get_snake_picks(st.session_state.draft_position)
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
                    if drafter_id == st.session_state.draft_position:
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
    
    # Main content - Add post-draft analysis tab when draft is complete
    draft_complete = len(st.session_state.drafted_teams) == 32
    
    # Show draft completion notification
    if draft_complete:
        st.success("🎉 **DRAFT COMPLETE!** All 32 teams have been drafted. Check out the Post-Draft Analysis tab for comprehensive insights!")
    
    if draft_complete:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Recommendations", "🏈 Team Rankings", "📈 Draft Board", 
            "🧠 Strategy", "🏆 Post-Draft Analysis"
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Recommendations", "🏈 Team Rankings", "📈 Draft Board", "🧠 Strategy"
        ])
    
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
                        'Drafter': f"{'YOU' if drafter_id == st.session_state.draft_position else f'#{drafter_id}'}",
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
        st.subheader(f"🎯 Position #{st.session_state.draft_position} Strategy")
        
        pick_advice = generate_pick_advice(st.session_state.draft_position)
        
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
    
    # Post-draft analysis tab (only shown when draft is complete)
    if draft_complete:
        with tab5:
            st.header("🏆 Post-Draft Analysis")
            
            # Initialize analyzer
            analyzer = PostDraftAnalyzer(team_data, n_simulations=10000)
            
            with st.spinner("Running post-draft analysis..."):
                # Calculate prize probabilities
                prize_probs = analyzer.calculate_prize_probabilities(st.session_state.drafter_teams)
                
                # Get user's performance
                user_teams = st.session_state.my_teams
                our_picks = get_snake_picks(st.session_state.draft_position)
                user_grade = analyzer.evaluate_draft_grade(user_teams, our_picks)
                
                # Strategy effectiveness analysis
                strategy_analysis = analyzer.analyze_strategy_effectiveness(st.session_state.drafter_teams)
                
                # Create pick history for best/worst picks
                pick_history = []
                for i, team in enumerate(st.session_state.drafted_teams, 1):
                    # Find which drafter picked this team
                    drafter_id = None
                    for did, teams in st.session_state.drafter_teams.items():
                        if team in teams:
                            drafter_id = did
                            break
                    
                    if drafter_id:
                        pick_history.append({
                            'team': team,
                            'pick_number': i,
                            'drafter_id': drafter_id
                        })
                
                best_worst_picks = analyzer.identify_best_worst_picks(pick_history)
                predicted_standings = analyzer.predict_final_standings(prize_probs)
            
            # Display results
            st.subheader("🎯 Your Draft Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                grade_colors = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🟠', 'F': '🔴'}
                st.metric("Draft Grade", 
                         f"{grade_colors.get(user_grade['grade'], '⚪')} {user_grade['grade']}")
            with col2:
                st.metric("Performance Ratio", f"{user_grade['performance_ratio']:.2f}")
            with col3:
                user_prob = prize_probs[prize_probs['drafter_id'] == st.session_state.draft_position]
                if not user_prob.empty:
                    st.metric("Win Prize Probability", f"{user_prob.iloc[0]['win_prize_prob']:.1%}")
            with col4:
                if not user_prob.empty:
                    st.metric("Loss Prize Probability", f"{user_prob.iloc[0]['loss_prize_prob']:.1%}")
            
            st.info(f"**Analysis:** {user_grade['analysis']}")
            
            # Prize probabilities table
            st.subheader("🏆 Prize Probabilities")
            
            display_probs = prize_probs[['drafter_id', 'expected_wins', 'expected_losses', 
                                        'win_prize_prob', 'loss_prize_prob']].copy()
            display_probs.columns = ['Drafter', 'Expected Wins', 'Expected Losses', 
                                    'Win Prize %', 'Loss Prize %']
            display_probs['Win Prize %'] = (display_probs['Win Prize %'] * 100).round(1)
            display_probs['Loss Prize %'] = (display_probs['Loss Prize %'] * 100).round(1)
            
            # Highlight user's row
            def highlight_user_row(row):
                return ['background-color: lightgreen' if row['Drafter'] == st.session_state.draft_position else '' for _ in row]
            
            styled_probs = display_probs.style.apply(highlight_user_row, axis=1)
            st.dataframe(styled_probs, use_container_width=True, hide_index=True)
            
            # Visualization - Prize probability scatter plot
            fig = px.scatter(display_probs, 
                           x='Win Prize %', 
                           y='Loss Prize %',
                           text='Drafter',
                           title="Prize Probability Matrix",
                           labels={'Win Prize %': 'Win Prize Probability (%)', 
                                  'Loss Prize %': 'Loss Prize Probability (%)'})
            
            # Highlight user's position
            fig.add_scatter(x=[display_probs[display_probs['Drafter'] == st.session_state.draft_position]['Win Prize %'].iloc[0] if not display_probs[display_probs['Drafter'] == st.session_state.draft_position].empty else 0],
                          y=[display_probs[display_probs['Drafter'] == st.session_state.draft_position]['Loss Prize %'].iloc[0] if not display_probs[display_probs['Drafter'] == st.session_state.draft_position].empty else 0],
                          mode='markers',
                          marker=dict(size=15, color='red', symbol='star'),
                          name='YOU',
                          showlegend=True)
            
            fig.update_traces(textposition='top center')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best and worst picks
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✨ Best Value Picks")
                if best_worst_picks['best_picks']:
                    best_df = pd.DataFrame(best_worst_picks['best_picks'])
                    best_df['value_captured'] = best_df['value_captured'].round(2)
                    st.dataframe(best_df[['team', 'pick_number', 'drafter', 'value_captured']], 
                               hide_index=True, use_container_width=True)
                else:
                    st.info("No picks analyzed yet")
            
            with col2:
                st.subheader("❌ Biggest Reaches")
                if best_worst_picks['worst_picks']:
                    worst_df = pd.DataFrame(best_worst_picks['worst_picks'])
                    worst_df['value_captured'] = worst_df['value_captured'].round(2)
                    st.dataframe(worst_df[['team', 'pick_number', 'drafter', 'value_captured']], 
                               hide_index=True, use_container_width=True)
                else:
                    st.info("No picks analyzed yet")
            
            # Strategy effectiveness
            st.subheader("📊 Strategy Effectiveness")
            
            strategy_display = strategy_analysis[['drafter_id', 'detected_strategy', 'avg_wins', 
                                                'strategy_consistency']].copy()
            strategy_display.columns = ['Drafter', 'Strategy', 'Avg Wins', 'Consistency Score']
            strategy_display['Avg Wins'] = strategy_display['Avg Wins'].round(1)
            strategy_display['Consistency Score'] = strategy_display['Consistency Score'].round(1)
            
            # Highlight user's strategy
            styled_strategy = strategy_display.style.apply(
                lambda row: ['background-color: lightgreen' if row['Drafter'] == st.session_state.draft_position else '' for _ in row], 
                axis=1
            )
            st.dataframe(styled_strategy, use_container_width=True, hide_index=True)
            
            # Final standings prediction
            st.subheader("🔮 Predicted Final Standings")
            
            standings_display = predicted_standings[['drafter_id', 'predicted_rank', 'championship_score', 
                                                   'confidence']].copy()
            standings_display.columns = ['Drafter', 'Predicted Rank', 'Championship Score', 'Confidence']
            standings_display['Championship Score'] = standings_display['Championship Score'].round(1)
            
            # Highlight user's predicted finish
            styled_standings = standings_display.style.apply(
                lambda row: ['background-color: lightgreen' if row['Drafter'] == st.session_state.draft_position else '' for _ in row], 
                axis=1
            )
            st.dataframe(styled_standings, use_container_width=True, hide_index=True)
            
            # Summary insights
            st.subheader("💡 Key Insights")
            
            user_standing = predicted_standings[predicted_standings['drafter_id'] == st.session_state.draft_position]
            if not user_standing.empty:
                rank = int(user_standing.iloc[0]['predicted_rank'])
                confidence = user_standing.iloc[0]['confidence']
                
                insights = []
                
                if rank <= 2:
                    insights.append("🏆 You're predicted to finish in the top 2! Strong draft execution.")
                elif rank <= 4:
                    insights.append("📈 Solid draft positioning. You're in contention for prizes.")
                else:
                    insights.append("🎯 Room for improvement, but anything can happen in the actual season.")
                
                if user_grade['grade'] in ['A', 'B']:
                    insights.append("✅ You captured good value relative to your draft position.")
                
                if confidence == 'HIGH':
                    insights.append("🔒 High confidence prediction - your strategy was clear and consistent.")
                elif confidence == 'LOW':
                    insights.append("🎲 Low confidence prediction - lots of variance in your portfolio.")
                
                for insight in insights:
                    st.success(insight)

if __name__ == "__main__":
    main()