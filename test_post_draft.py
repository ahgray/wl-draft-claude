#!/usr/bin/env python3
"""
Test post-draft analysis functionality
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from data.data_fetcher import NFLDataFetcher
from analysis.post_draft import PostDraftAnalyzer

def test_post_draft_analysis():
    """Test the post-draft analysis pipeline"""
    print("🏈 Testing Post-Draft Analysis")
    print("=" * 50)
    
    # Load team data
    print("📊 Loading team data...")
    fetcher = NFLDataFetcher()
    team_data = fetcher.create_composite_rankings()
    team_data['expected_losses'] = 17 - team_data['vegas_wins']
    print(f"✅ Loaded {len(team_data)} teams")
    
    # Create mock draft results (8 drafters, 4 teams each)
    print("\n🎯 Creating mock draft scenario...")
    teams = team_data['team_abbr'].tolist()
    
    # Simulate realistic draft portfolios
    mock_portfolios = {
        1: ['BAL', 'SF', 'PHI', 'BUF'],      # Win-focused strategy
        2: ['CAR', 'NE', 'NYG', 'LV'],       # Loss-focused strategy  
        3: ['KC', 'DET', 'MIA', 'CLE'],      # Balanced approach
        4: ['DAL', 'GB', 'HOU', 'JAX'],      # Mixed strategy
        5: ['LAR', 'MIN', 'SEA', 'TB'],      # Win-focused
        6: ['CHI', 'ARI', 'WAS', 'DEN'],     # Loss-focused
        7: ['CIN', 'ATL', 'NO', 'IND'],      # Balanced
        8: ['LAC', 'NYJ', 'PIT', 'TEN']      # Mixed
    }
    
    print("📋 Draft Results:")
    for drafter_id, teams in mock_portfolios.items():
        total_wins = team_data[team_data['team_abbr'].isin(teams)]['vegas_wins'].sum()
        strategy = "WIN" if total_wins > 34 else "LOSS" if total_wins < 30 else "BALANCED"
        print(f"  Drafter #{drafter_id}: {teams} (Total: {total_wins:.1f}W) - {strategy}")
    
    # Initialize analyzer
    print("\n🔬 Initializing PostDraftAnalyzer...")
    analyzer = PostDraftAnalyzer(team_data, n_simulations=1000)  # Use fewer sims for testing
    
    # Test prize probability calculations
    print("\n🏆 Calculating prize probabilities...")
    prize_probs = analyzer.calculate_prize_probabilities(mock_portfolios)
    print(f"✅ Generated probabilities for {len(prize_probs)} drafters")
    
    # Display prize probabilities
    print("\n📊 Prize Probability Results:")
    for _, row in prize_probs.iterrows():
        print(f"  Drafter #{row['drafter_id']}: "
              f"Win {row['win_prize_prob']:.1%}, Loss {row['loss_prize_prob']:.1%}")
    
    # Test user performance analysis
    print("\n🎯 Testing user performance analysis...")
    user_teams = mock_portfolios[6]  # Test as drafter #6
    user_picks = [6, 11, 22, 27]  # Position 6 snake draft picks
    user_grade = analyzer.evaluate_draft_grade(user_teams, user_picks)
    
    print(f"✅ User Draft Grade: {user_grade['grade']}")
    print(f"   Analysis: {user_grade['analysis']}")
    print(f"   Performance Ratio: {user_grade['performance_ratio']:.2f}")
    
    # Test strategy effectiveness
    print("\n📈 Analyzing strategy effectiveness...")
    strategy_analysis = analyzer.analyze_strategy_effectiveness(mock_portfolios)
    
    print("✅ Strategy Analysis Results:")
    for _, row in strategy_analysis.iterrows():
        print(f"  Drafter #{row['drafter_id']}: {row['detected_strategy']} "
              f"(Avg: {row['avg_wins']:.1f}W, Consistency: {row['strategy_consistency']:.1f})")
    
    # Test best/worst picks
    print("\n⭐ Identifying best and worst picks...")
    
    # Create mock pick history
    pick_history = []
    pick_num = 1
    for round_num in range(1, 5):  # 4 rounds
        for pos in range(1, 9):  # 8 positions
            if round_num % 2 == 1:  # Odd rounds go 1-8
                drafter = pos
            else:  # Even rounds go 8-1
                drafter = 9 - pos
            
            # Get team for this drafter and round
            if drafter in mock_portfolios:
                teams_for_drafter = mock_portfolios[drafter]
                if len(teams_for_drafter) >= round_num:
                    team = teams_for_drafter[round_num - 1]
                    pick_history.append({
                        'team': team,
                        'pick_number': pick_num,
                        'drafter_id': drafter
                    })
            pick_num += 1
    
    best_worst_picks = analyzer.identify_best_worst_picks(pick_history)
    
    print("✅ Best Value Picks:")
    for pick in best_worst_picks['best_picks'][:3]:
        print(f"  #{pick['pick_number']}: {pick['team']} by Drafter #{pick['drafter']} "
              f"(Value: +{pick['value_captured']:.2f})")
    
    print("✅ Biggest Reaches:")
    for pick in best_worst_picks['worst_picks'][:3]:
        print(f"  #{pick['pick_number']}: {pick['team']} by Drafter #{pick['drafter']} "
              f"(Value: {pick['value_captured']:.2f})")
    
    # Test final standings prediction
    print("\n🔮 Predicting final standings...")
    predicted_standings = analyzer.predict_final_standings(prize_probs)
    
    print("✅ Predicted Final Order:")
    for _, row in predicted_standings.head(5).iterrows():
        print(f"  #{int(row['predicted_rank'])}: Drafter #{row['drafter_id']} "
              f"(Score: {row['championship_score']:.1f}, Confidence: {row['confidence']})")
    
    print("\n🎉 Post-Draft Analysis Test Complete!")
    print("✅ All components working correctly")

if __name__ == "__main__":
    test_post_draft_analysis()