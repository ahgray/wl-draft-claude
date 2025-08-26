"""
Win Probability Model for NFL Games
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats

class WinProbabilityModel:
    def __init__(self, 
                 home_advantage: float = 2.65,
                 k_factor: float = 0.04,
                 use_weather: bool = False):
        """
        Initialize win probability model
        
        Args:
            home_advantage: Points added for home team (historically 2.65)
            k_factor: Scaling factor for logistic function (calibrated to 0.04)
            use_weather: Whether to adjust for weather conditions
        """
        self.home_advantage = home_advantage
        self.k_factor = k_factor
        self.use_weather = use_weather
        
    def calculate_win_probability(self, 
                                 team_a_rating: float,
                                 team_b_rating: float,
                                 is_home_a: bool = True,
                                 neutral_site: bool = False) -> float:
        """
        Calculate probability of team A winning against team B
        
        Uses logistic regression model:
        P(A wins) = 1 / (1 + exp(-k * rating_diff))
        """
        # Apply home field advantage
        if neutral_site:
            home_adj = 0
        elif is_home_a:
            home_adj = self.home_advantage
        else:
            home_adj = -self.home_advantage
            
        # Calculate rating differential
        rating_diff = team_a_rating - team_b_rating + home_adj
        
        # Logistic function
        win_prob = 1 / (1 + np.exp(-self.k_factor * rating_diff))
        
        return win_prob
    
    def simulate_game(self,
                     team_a_rating: float,
                     team_b_rating: float,
                     is_home_a: bool = True,
                     rating_std: float = 2.5) -> bool:
        """
        Simulate a single game with uncertainty
        
        Returns:
            True if team A wins, False if team B wins
        """
        # Add random variation to ratings (represents game-day uncertainty)
        team_a_performance = np.random.normal(team_a_rating, rating_std)
        team_b_performance = np.random.normal(team_b_rating, rating_std)
        
        # Calculate win probability with actual performance
        win_prob = self.calculate_win_probability(
            team_a_performance, 
            team_b_performance,
            is_home_a
        )
        
        # Simulate outcome
        return np.random.random() < win_prob
    
    def calculate_season_win_distribution(self,
                                         team: str,
                                         schedule: pd.DataFrame,
                                         ratings: Dict[str, float],
                                         n_simulations: int = 10000) -> np.ndarray:
        """
        Calculate win distribution for a team's season
        
        Returns:
            Array of win totals from simulations
        """
        # Get team's games
        home_games = schedule[schedule['home_team'] == team]
        away_games = schedule[schedule['away_team'] == team]
        
        wins = np.zeros(n_simulations)
        
        for sim in range(n_simulations):
            season_wins = 0
            
            # Simulate home games
            for _, game in home_games.iterrows():
                opponent = game['away_team']
                if self.simulate_game(ratings[team], ratings[opponent], is_home_a=True):
                    season_wins += 1
            
            # Simulate away games
            for _, game in away_games.iterrows():
                opponent = game['home_team']
                if self.simulate_game(ratings[team], ratings[opponent], is_home_a=False):
                    season_wins += 1
            
            wins[sim] = season_wins
        
        return wins
    
    def calculate_expected_wins(self,
                               team: str,
                               schedule: pd.DataFrame,
                               ratings: Dict[str, float]) -> float:
        """
        Calculate expected wins for a team (no simulation, just probabilities)
        """
        expected_wins = 0.0
        
        # Home games
        home_games = schedule[schedule['home_team'] == team]
        for _, game in home_games.iterrows():
            opponent = game['away_team']
            if opponent in ratings:
                win_prob = self.calculate_win_probability(
                    ratings[team], ratings[opponent], is_home_a=True
                )
                expected_wins += win_prob
        
        # Away games
        away_games = schedule[schedule['away_team'] == team]
        for _, game in away_games.iterrows():
            opponent = game['home_team']
            if opponent in ratings:
                win_prob = self.calculate_win_probability(
                    ratings[team], ratings[opponent], is_home_a=False
                )
                expected_wins += win_prob
        
        return expected_wins
    
    def calibrate_k_factor(self,
                          historical_games: pd.DataFrame,
                          ratings: Dict[str, float]) -> float:
        """
        Calibrate k-factor using historical data
        Uses maximum likelihood estimation
        """
        from scipy.optimize import minimize_scalar
        
        def neg_log_likelihood(k):
            """Negative log-likelihood for optimization"""
            self.k_factor = k
            log_likelihood = 0
            
            for _, game in historical_games.iterrows():
                home = game['home_team']
                away = game['away_team']
                home_won = game['home_won']
                
                if home in ratings and away in ratings:
                    win_prob = self.calculate_win_probability(
                        ratings[home], ratings[away], is_home_a=True
                    )
                    
                    # Add to log-likelihood
                    if home_won:
                        log_likelihood += np.log(win_prob + 1e-10)
                    else:
                        log_likelihood += np.log(1 - win_prob + 1e-10)
            
            return -log_likelihood
        
        # Optimize k-factor
        result = minimize_scalar(neg_log_likelihood, bounds=(0.01, 0.1), method='bounded')
        optimal_k = result.x
        
        self.k_factor = optimal_k
        print(f"Calibrated k-factor: {optimal_k:.4f}")
        
        return optimal_k
    
    def calculate_correlation_matrix(self,
                                    teams: list,
                                    schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation between teams based on shared opponents
        """
        n_teams = len(teams)
        correlation = np.zeros((n_teams, n_teams))
        
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i == j:
                    correlation[i, j] = 1.0
                else:
                    # Calculate shared opponents
                    team1_opponents = set()
                    team1_opponents.update(schedule[schedule['home_team'] == team1]['away_team'])
                    team1_opponents.update(schedule[schedule['away_team'] == team1]['home_team'])
                    
                    team2_opponents = set()
                    team2_opponents.update(schedule[schedule['home_team'] == team2]['away_team'])
                    team2_opponents.update(schedule[schedule['away_team'] == team2]['home_team'])
                    
                    # Shared opponents ratio
                    shared = len(team1_opponents & team2_opponents)
                    total = len(team1_opponents | team2_opponents)
                    
                    if total > 0:
                        correlation[i, j] = shared / total
                    
                    # Extra correlation if they play each other
                    if team2 in team1_opponents:
                        correlation[i, j] += 0.2
        
        return pd.DataFrame(correlation, index=teams, columns=teams)


class DivisionAnalyzer:
    """Analyze divisional dynamics for win/loss optimization"""
    
    def __init__(self):
        self.divisions = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['KC', 'LAC', 'DEN', 'LV'],
            'NFC East': ['PHI', 'DAL', 'WAS', 'NYG'],
            'NFC North': ['DET', 'GB', 'MIN', 'CHI'],
            'NFC South': ['TB', 'NO', 'ATL', 'CAR'],
            'NFC West': ['SF', 'LAR', 'SEA', 'ARI']
        }
    
    def get_division(self, team: str) -> str:
        """Get division for a team"""
        for div, teams in self.divisions.items():
            if team in teams:
                return div
        return None
    
    def calculate_division_games(self, team: str) -> int:
        """Calculate number of division games (usually 6)"""
        return 6
    
    def identify_division_cannibalization(self, portfolio: list) -> Dict:
        """Identify if portfolio has teams from same division"""
        cannibalization = {}
        
        for i, team1 in enumerate(portfolio):
            div1 = self.get_division(team1)
            for team2 in portfolio[i+1:]:
                div2 = self.get_division(team2)
                if div1 == div2:
                    key = f"{team1}-{team2}"
                    cannibalization[key] = {
                        'division': div1,
                        'games': 2,  # Teams play twice in division
                        'impact': 'high'
                    }
        
        return cannibalization