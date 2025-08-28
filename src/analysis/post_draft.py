"""
Post-Draft Analysis Module
Comprehensive analysis after draft completion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

class PostDraftAnalyzer:
    def __init__(self, team_data: pd.DataFrame, n_simulations: int = 10000):
        """
        Initialize post-draft analyzer
        
        Args:
            team_data: DataFrame with team statistics
            n_simulations: Number of Monte Carlo simulations to run
        """
        self.team_data = team_data
        self.n_simulations = n_simulations
        
    def calculate_prize_probabilities(self, all_portfolios: Dict[int, List[str]]) -> pd.DataFrame:
        """
        Calculate each drafter's probability of winning each prize
        
        Returns:
            DataFrame with win/loss prize probabilities for each drafter
        """
        results = []
        
        # Run simulations for each portfolio
        for drafter_id, teams in all_portfolios.items():
            if not teams:
                continue
                
            portfolio_data = self.team_data[self.team_data['team_abbr'].isin(teams)]
            
            # Simple simulation using expected wins + variance
            simulated_wins = []
            simulated_losses = []
            
            for _ in range(self.n_simulations):
                # Add random variance to expected wins
                total_wins = 0
                for _, team in portfolio_data.iterrows():
                    team_wins = np.random.normal(
                        team.get('vegas_wins', 8.5),
                        2.5  # Standard deviation
                    )
                    team_wins = np.clip(team_wins, 0, 17)
                    total_wins += team_wins
                
                simulated_wins.append(total_wins)
                simulated_losses.append(len(teams) * 17 - total_wins)
            
            results.append({
                'drafter_id': drafter_id,
                'teams': teams,
                'expected_wins': np.mean(simulated_wins),
                'expected_losses': np.mean(simulated_losses),
                'std_wins': np.std(simulated_wins),
                'std_losses': np.std(simulated_losses),
                'p90_wins': np.percentile(simulated_wins, 90),
                'p10_wins': np.percentile(simulated_wins, 10),
                'p90_losses': np.percentile(simulated_losses, 90),
                'p10_losses': np.percentile(simulated_losses, 10),
                'simulated_wins': simulated_wins,
                'simulated_losses': simulated_losses
            })
        
        # Calculate prize probabilities
        df = pd.DataFrame(results)
        
        # For each simulation, determine winner
        win_prize_counts = {d: 0 for d in all_portfolios.keys()}
        loss_prize_counts = {d: 0 for d in all_portfolios.keys()}
        
        for sim_idx in range(self.n_simulations):
            sim_results = []
            for _, row in df.iterrows():
                sim_results.append({
                    'drafter_id': row['drafter_id'],
                    'wins': row['simulated_wins'][sim_idx],
                    'losses': row['simulated_losses'][sim_idx]
                })
            
            sim_df = pd.DataFrame(sim_results)
            
            # Find winners
            max_wins = sim_df['wins'].max()
            max_losses = sim_df['losses'].max()
            
            win_winners = sim_df[sim_df['wins'] == max_wins]['drafter_id'].tolist()
            loss_winners = sim_df[sim_df['losses'] == max_losses]['drafter_id'].tolist()
            
            # Split prize if tied
            for winner in win_winners:
                win_prize_counts[winner] += 1 / len(win_winners)
            for winner in loss_winners:
                loss_prize_counts[winner] += 1 / len(loss_winners)
        
        # Add probabilities to dataframe
        df['win_prize_prob'] = df['drafter_id'].map(lambda x: win_prize_counts[x] / self.n_simulations)
        df['loss_prize_prob'] = df['drafter_id'].map(lambda x: loss_prize_counts[x] / self.n_simulations)
        
        return df
    
    def evaluate_draft_grade(self, my_teams: List[str], my_pick_positions: List[int]) -> Dict:
        """
        Grade the user's draft performance
        
        Returns:
            Dictionary with grade and analysis
        """
        if not my_teams:
            return {'grade': 'N/A', 'analysis': 'No teams drafted'}
        
        my_team_data = self.team_data[self.team_data['team_abbr'].isin(my_teams)]
        
        # Calculate value captured
        total_rating = my_team_data['composite_rating'].sum()
        avg_rating = my_team_data['composite_rating'].mean()
        
        # Expected value at pick positions
        sorted_teams = self.team_data.sort_values('composite_rating', ascending=False)
        expected_ratings = []
        for pick_pos in my_pick_positions:
            if pick_pos <= len(sorted_teams):
                expected_ratings.append(sorted_teams.iloc[pick_pos - 1]['composite_rating'])
        
        expected_avg = np.mean(expected_ratings) if expected_ratings else 0
        
        # Calculate grade based on actual vs expected
        performance_ratio = avg_rating / expected_avg if expected_avg > 0 else 1
        
        if performance_ratio >= 1.1:
            grade = 'A'
            analysis = "Excellent value! Significantly outperformed expected picks."
        elif performance_ratio >= 1.0:
            grade = 'B'
            analysis = "Good draft. Captured expected value or better."
        elif performance_ratio >= 0.9:
            grade = 'C'
            analysis = "Average draft. Slightly below expected value."
        elif performance_ratio >= 0.8:
            grade = 'D'
            analysis = "Below average. Missed some value opportunities."
        else:
            grade = 'F'
            analysis = "Poor draft. Significant value left on the table."
        
        return {
            'grade': grade,
            'analysis': analysis,
            'actual_avg_rating': avg_rating,
            'expected_avg_rating': expected_avg,
            'performance_ratio': performance_ratio,
            'total_rating': total_rating
        }
    
    def analyze_strategy_effectiveness(self, all_portfolios: Dict[int, List[str]]) -> pd.DataFrame:
        """
        Analyze how effectively each drafter executed their strategy
        """
        results = []
        
        for drafter_id, teams in all_portfolios.items():
            if not teams:
                continue
            
            portfolio_data = self.team_data[self.team_data['team_abbr'].isin(teams)]
            avg_wins = portfolio_data['vegas_wins'].mean() if not portfolio_data.empty else 8.5
            
            # Detect strategy
            if avg_wins > 9.5:
                strategy = 'WIN_FOCUSED'
                effectiveness = portfolio_data['vegas_wins'].std()  # Lower std = more consistent
                effectiveness_score = 100 * (1 - effectiveness / 3)  # Normalize
            elif avg_wins < 7.5:
                strategy = 'LOSS_FOCUSED'
                effectiveness = portfolio_data['expected_losses'].std() if 'expected_losses' in portfolio_data else 2
                effectiveness_score = 100 * (1 - effectiveness / 3)
            else:
                strategy = 'BALANCED'
                # For balanced, look at variance
                effectiveness_score = 50 + (8.5 - abs(avg_wins - 8.5)) * 10
            
            results.append({
                'drafter_id': drafter_id,
                'detected_strategy': strategy,
                'avg_wins': avg_wins,
                'strategy_consistency': effectiveness_score,
                'team_quality_std': portfolio_data['composite_rating'].std() if not portfolio_data.empty else 0
            })
        
        return pd.DataFrame(results)
    
    def identify_best_worst_picks(self, pick_history: List[Dict]) -> Dict:
        """
        Identify the best and worst picks of the draft
        """
        pick_values = []
        
        for pick in pick_history:
            team = pick.get('team')
            pick_number = pick.get('pick_number')
            
            if team and pick_number:
                team_data = self.team_data[self.team_data['team_abbr'] == team]
                if not team_data.empty:
                    team_rating = team_data.iloc[0]['composite_rating']
                    
                    # Expected rating at this pick position
                    sorted_teams = self.team_data.sort_values('composite_rating', ascending=False)
                    expected_rating = sorted_teams.iloc[pick_number - 1]['composite_rating'] if pick_number <= len(sorted_teams) else 0
                    
                    value = team_rating - expected_rating
                    
                    pick_values.append({
                        'team': team,
                        'pick_number': pick_number,
                        'drafter': pick.get('drafter_id'),
                        'actual_rating': team_rating,
                        'expected_rating': expected_rating,
                        'value_captured': value
                    })
        
        if not pick_values:
            return {'best_picks': [], 'worst_picks': []}
        
        df = pd.DataFrame(pick_values)
        
        return {
            'best_picks': df.nlargest(5, 'value_captured').to_dict('records'),
            'worst_picks': df.nsmallest(5, 'value_captured').to_dict('records'),
            'all_picks': df
        }
    
    def predict_final_standings(self, prize_probabilities: pd.DataFrame) -> pd.DataFrame:
        """
        Predict final standings with confidence levels
        """
        standings = prize_probabilities.copy()
        
        # Calculate combined score
        standings['championship_score'] = (
            standings['win_prize_prob'] * 100 + 
            standings['loss_prize_prob'] * 100
        )
        
        # Rank by most likely to win something
        standings['predicted_rank'] = standings['championship_score'].rank(ascending=False)
        
        # Add confidence levels
        standings['confidence'] = standings.apply(
            lambda row: 'HIGH' if max(row['win_prize_prob'], row['loss_prize_prob']) > 0.3
            else 'MEDIUM' if max(row['win_prize_prob'], row['loss_prize_prob']) > 0.15
            else 'LOW', axis=1
        )
        
        return standings.sort_values('predicted_rank')
    
    def calculate_head_to_head_impact(self, all_portfolios: Dict[int, List[str]]) -> pd.DataFrame:
        """
        Calculate how much portfolios affect each other through head-to-head games
        """
        h2h_matrix = []
        
        for drafter1, teams1 in all_portfolios.items():
            for drafter2, teams2 in all_portfolios.items():
                if drafter1 != drafter2:
                    # Count how many times teams play each other
                    h2h_games = 0
                    for team1 in teams1:
                        for team2 in teams2:
                            # In reality, check schedule
                            # For now, assume division rivals play twice, others once
                            h2h_games += 1 if team1 != team2 else 0
                    
                    h2h_matrix.append({
                        'drafter1': drafter1,
                        'drafter2': drafter2,
                        'h2h_games': h2h_games,
                        'impact_score': h2h_games / (len(teams1) * 17) if teams1 else 0
                    })
        
        return pd.DataFrame(h2h_matrix)