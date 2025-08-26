"""
Monte Carlo Simulation Engine for NFL Season
Optimized for speed with vectorized operations
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class MonteCarloSimulator:
    def __init__(self, 
                 n_simulations: int = 100000,
                 use_parallel: bool = True):
        """
        Initialize Monte Carlo simulator
        
        Args:
            n_simulations: Number of season simulations to run
            use_parallel: Use multiprocessing for speed
        """
        self.n_simulations = n_simulations
        self.use_parallel = use_parallel
        self.n_cores = mp.cpu_count() - 1 if use_parallel else 1
        
    def simulate_season_vectorized(self,
                                  schedule: pd.DataFrame,
                                  team_ratings: Dict[str, float],
                                  rating_std: float = 2.5,
                                  home_advantage: float = 2.65,
                                  k_factor: float = 0.04) -> np.ndarray:
        """
        Vectorized season simulation for maximum speed
        
        Returns:
            Array of shape (n_teams, n_simulations) with win totals
        """
        teams = list(team_ratings.keys())
        n_teams = len(teams)
        team_idx = {team: i for i, team in enumerate(teams)}
        
        # Initialize wins array
        wins = np.zeros((n_teams, self.n_simulations))
        
        # Convert schedule to numpy arrays for speed
        home_teams = schedule['home_team'].values
        away_teams = schedule['away_team'].values
        
        # Vectorize all games at once
        for game_idx in range(len(schedule)):
            home = home_teams[game_idx]
            away = away_teams[game_idx]
            
            if home in team_idx and away in team_idx:
                home_idx = team_idx[home]
                away_idx = team_idx[away]
                
                # Generate random performances for all simulations at once
                home_performance = np.random.normal(
                    team_ratings[home] + home_advantage, 
                    rating_std, 
                    self.n_simulations
                )
                away_performance = np.random.normal(
                    team_ratings[away], 
                    rating_std, 
                    self.n_simulations
                )
                
                # Calculate win probabilities
                rating_diff = home_performance - away_performance
                home_win_prob = 1 / (1 + np.exp(-k_factor * rating_diff))
                
                # Simulate outcomes
                home_wins_game = np.random.random(self.n_simulations) < home_win_prob
                
                # Update win totals
                wins[home_idx] += home_wins_game
                wins[away_idx] += ~home_wins_game
        
        return wins, teams
    
    def calculate_team_distributions(self, 
                                    wins_array: np.ndarray,
                                    teams: list) -> pd.DataFrame:
        """
        Calculate win/loss distributions and statistics for all teams
        """
        results = []
        
        for i, team in enumerate(teams):
            team_wins = wins_array[i]
            team_losses = 17 - team_wins  # 17 game season
            
            results.append({
                'team': team,
                'mean_wins': np.mean(team_wins),
                'median_wins': np.median(team_wins),
                'std_wins': np.std(team_wins),
                'min_wins': np.min(team_wins),
                'max_wins': np.max(team_wins),
                'p10_wins': np.percentile(team_wins, 10),
                'p90_wins': np.percentile(team_wins, 90),
                'mean_losses': np.mean(team_losses),
                'p10_losses': np.percentile(team_losses, 10),
                'p90_losses': np.percentile(team_losses, 90),
                'win_variance': np.var(team_wins),
                'playoff_prob': np.mean(team_wins >= 10),  # Rough playoff threshold
                'top_pick_prob': np.mean(team_wins <= 3),  # Top draft pick threshold
                'win_distribution': team_wins  # Store full distribution
            })
        
        return pd.DataFrame(results)
    
    def calculate_portfolio_outcomes(self,
                                    portfolio: list,
                                    team_distributions: pd.DataFrame) -> Dict:
        """
        Calculate outcomes for a 4-team portfolio
        """
        # Get distributions for portfolio teams
        portfolio_dists = team_distributions[team_distributions['team'].isin(portfolio)]
        
        # Sum wins across portfolio for each simulation
        portfolio_wins = np.zeros(self.n_simulations)
        portfolio_losses = np.zeros(self.n_simulations)
        
        for _, team_data in portfolio_dists.iterrows():
            team_wins = team_data['win_distribution']
            portfolio_wins += team_wins
            portfolio_losses += (17 - team_wins)
        
        return {
            'portfolio': portfolio,
            'mean_wins': np.mean(portfolio_wins),
            'mean_losses': np.mean(portfolio_losses),
            'std_wins': np.std(portfolio_wins),
            'std_losses': np.std(portfolio_losses),
            'p10_wins': np.percentile(portfolio_wins, 10),
            'p90_wins': np.percentile(portfolio_wins, 90),
            'p10_losses': np.percentile(portfolio_losses, 10),
            'p90_losses': np.percentile(portfolio_losses, 90),
            'max_wins': np.max(portfolio_wins),
            'min_wins': np.min(portfolio_wins),
            'max_losses': np.max(portfolio_losses),
            'min_losses': np.min(portfolio_losses),
            'win_distribution': portfolio_wins,
            'loss_distribution': portfolio_losses
        }
    
    def identify_correlations(self,
                            portfolio: list,
                            schedule: pd.DataFrame) -> Dict:
        """
        Identify correlations and cannibalization in portfolio
        """
        correlations = {
            'head_to_head_games': 0,
            'shared_opponents': set(),
            'division_conflicts': []
        }
        
        # Check head-to-head games
        for i, team1 in enumerate(portfolio):
            for team2 in portfolio[i+1:]:
                h2h_games = len(schedule[
                    ((schedule['home_team'] == team1) & (schedule['away_team'] == team2)) |
                    ((schedule['home_team'] == team2) & (schedule['away_team'] == team1))
                ])
                correlations['head_to_head_games'] += h2h_games
        
        # Check shared opponents
        for team in portfolio:
            opponents = set()
            opponents.update(schedule[schedule['home_team'] == team]['away_team'])
            opponents.update(schedule[schedule['away_team'] == team]['home_team'])
            
            for opp in opponents:
                if opp not in portfolio:
                    correlations['shared_opponents'].add(opp)
        
        return correlations
    
    def run_full_simulation(self,
                          schedule: pd.DataFrame,
                          team_ratings: Dict[str, float],
                          cache_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run complete simulation and optionally cache results
        """
        print(f"Running {self.n_simulations:,} season simulations...")
        
        # Check cache first
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached simulation from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Run simulation
        wins_array, teams = self.simulate_season_vectorized(schedule, team_ratings)
        
        # Calculate distributions
        team_distributions = self.calculate_team_distributions(wins_array, teams)
        
        # Cache if requested
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(team_distributions, f)
            print(f"Cached simulation results to {cache_path}")
        
        return team_distributions
    
    def evaluate_all_portfolios(self,
                              team_distributions: pd.DataFrame,
                              max_combinations: int = 10000) -> pd.DataFrame:
        """
        Pre-evaluate portfolio combinations for quick lookup during draft
        """
        from itertools import combinations
        
        teams = team_distributions['team'].tolist()
        all_portfolios = list(combinations(teams, 4))
        
        # Limit if too many combinations
        if len(all_portfolios) > max_combinations:
            # Sample strategically - include best/worst teams
            np.random.seed(42)
            sampled_portfolios = np.random.choice(
                len(all_portfolios), 
                max_combinations, 
                replace=False
            )
            all_portfolios = [all_portfolios[i] for i in sampled_portfolios]
        
        print(f"Evaluating {len(all_portfolios)} portfolio combinations...")
        
        portfolio_results = []
        for portfolio in tqdm(all_portfolios):
            outcome = self.calculate_portfolio_outcomes(portfolio, team_distributions)
            portfolio_results.append({
                'portfolio': portfolio,
                'mean_wins': outcome['mean_wins'],
                'mean_losses': outcome['mean_losses'],
                'win_score': outcome['p90_wins'],  # Upside for wins
                'loss_score': outcome['p90_losses'],  # Upside for losses
                'win_consistency': 1 / (outcome['std_wins'] + 1),
                'loss_consistency': 1 / (outcome['std_losses'] + 1)
            })
        
        return pd.DataFrame(portfolio_results)


class SimulationAnalyzer:
    """Analyze simulation results for insights"""
    
    def __init__(self, simulation_results: pd.DataFrame):
        self.results = simulation_results
    
    def create_tiers(self, metric: str = 'mean_wins', n_tiers: int = 5) -> Dict:
        """Create team tiers based on simulation results"""
        sorted_teams = self.results.sort_values(metric, ascending=False)
        tier_size = len(sorted_teams) // n_tiers
        
        tiers = {}
        for i in range(n_tiers):
            start_idx = i * tier_size
            end_idx = start_idx + tier_size if i < n_tiers - 1 else len(sorted_teams)
            tier_name = f"Tier_{i+1}"
            tiers[tier_name] = sorted_teams.iloc[start_idx:end_idx]['team'].tolist()
        
        return tiers
    
    def find_value_teams(self) -> pd.DataFrame:
        """Identify teams with high variance (good for chaos strategy)"""
        self.results['chaos_score'] = (
            self.results['std_wins'] * 
            (self.results['playoff_prob'] * self.results['top_pick_prob'])
        )
        
        value_teams = self.results.nlargest(10, 'chaos_score')[
            ['team', 'mean_wins', 'std_wins', 'chaos_score', 'playoff_prob', 'top_pick_prob']
        ]
        
        return value_teams
    
    def calculate_reach_probability(self, 
                                   team: str, 
                                   current_pick: int,
                                   next_pick: int,
                                   draft_position: int = 6) -> float:
        """
        Calculate probability a team will still be available at next pick
        """
        team_data = self.results[self.results['team'] == team].iloc[0]
        team_rank = self.results['mean_wins'].rank(ascending=False)[
            self.results['team'] == team
        ].values[0]
        
        picks_between = next_pick - current_pick - 1
        
        # Simple model: probability decreases with team quality and picks between
        base_prob = 1.0 - (team_rank / 32)
        decay_rate = 0.9
        reach_prob = base_prob * (decay_rate ** picks_between)
        
        return min(max(reach_prob, 0), 1)