"""
Draft Strategy Engine with Position #6 Optimizations
Handles real-time strategy adaptation and opponent modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Strategy(Enum):
    WIN_MAXIMIZER = "win_maximizer"
    LOSS_MAXIMIZER = "loss_maximizer"
    BALANCED = "balanced"
    CHAOS = "chaos"
    BLOCKING = "blocking"
    PIVOT = "pivot"

@dataclass
class DraftState:
    """Current state of the draft"""
    round: int
    pick_number: int
    available_teams: List[str]
    my_teams: List[str]
    opponent_teams: Dict[int, List[str]]  # Drafter ID -> teams
    current_strategy: Strategy
    
    @property
    def my_next_pick(self) -> int:
        """Calculate next pick number for position 6"""
        if self.round == 1:
            return 11  # Round 2
        elif self.round == 2:
            return 22  # Round 3
        elif self.round == 3:
            return 27  # Round 4
        else:
            return None
    
    @property
    def picks_until_next(self) -> int:
        """Picks until our next selection"""
        if self.my_next_pick:
            return self.my_next_pick - self.pick_number - 1
        return 0

class DraftStrategyEngine:
    def __init__(self, 
                 draft_position: int = 6,
                 total_drafters: int = 8,
                 teams_per_drafter: int = 4):
        """
        Initialize draft strategy engine
        
        Args:
            draft_position: Our draft position (default 6)
            total_drafters: Total number of drafters
            teams_per_drafter: Teams each drafter selects
        """
        self.draft_position = draft_position
        self.total_drafters = total_drafters
        self.teams_per_drafter = teams_per_drafter
        self.snake_picks = self._calculate_snake_picks()
        
    def _calculate_snake_picks(self) -> List[int]:
        """Calculate our pick numbers in snake draft"""
        picks = []
        for round_num in range(1, self.teams_per_drafter + 1):
            if round_num % 2 == 1:  # Odd rounds go forward
                pick = (round_num - 1) * self.total_drafters + self.draft_position
            else:  # Even rounds go backward
                pick = round_num * self.total_drafters - self.draft_position + 1
            picks.append(pick)
        return picks
    
    def recommend_strategy(self, 
                          draft_state: DraftState,
                          team_stats: pd.DataFrame) -> Strategy:
        """
        Recommend strategy based on current draft state
        """
        round_num = draft_state.round
        my_teams = draft_state.my_teams
        
        # Round 1: Flexible, see what's available
        if round_num == 1:
            # Check if elite teams available
            top_teams = team_stats.nlargest(5, 'mean_wins')['team'].tolist()
            bottom_teams = team_stats.nsmallest(5, 'mean_wins')['team'].tolist()
            
            available_top = [t for t in top_teams if t in draft_state.available_teams]
            available_bottom = [t for t in bottom_teams if t in draft_state.available_teams]
            
            if len(available_top) >= 2:
                return Strategy.WIN_MAXIMIZER
            elif len(available_bottom) >= 2:
                return Strategy.LOSS_MAXIMIZER
            else:
                return Strategy.BALANCED
        
        # Round 2: Commit to direction
        elif round_num == 2:
            if len(my_teams) == 1:
                first_pick_wins = team_stats[team_stats['team'] == my_teams[0]]['mean_wins'].values[0]
                if first_pick_wins > 9:
                    return Strategy.WIN_MAXIMIZER
                elif first_pick_wins < 7:
                    return Strategy.LOSS_MAXIMIZER
                else:
                    return Strategy.CHAOS
        
        # Round 3: Consider blocking
        elif round_num == 3:
            # Check if any opponent is dominating
            for drafter_id, opp_teams in draft_state.opponent_teams.items():
                if len(opp_teams) >= 2:
                    opp_wins = team_stats[team_stats['team'].isin(opp_teams)]['mean_wins'].sum()
                    if opp_wins > 20 or opp_wins < 14:
                        return Strategy.BLOCKING
            
            return draft_state.current_strategy
        
        # Round 4: Complete portfolio
        else:
            return draft_state.current_strategy
    
    def calculate_team_value(self,
                           team: str,
                           draft_state: DraftState,
                           team_stats: pd.DataFrame,
                           strategy: Strategy) -> float:
        """
        Calculate value of a team given current portfolio and strategy
        """
        team_data = team_stats[team_stats['team'] == team].iloc[0]
        base_value = 0
        
        # Base value depends on strategy
        if strategy == Strategy.WIN_MAXIMIZER:
            base_value = team_data['mean_wins'] * 2 + team_data['p90_wins']
        elif strategy == Strategy.LOSS_MAXIMIZER:
            base_value = team_data['mean_losses'] * 2 + team_data['p90_losses']
        elif strategy == Strategy.CHAOS:
            base_value = team_data['std_wins'] * team_data['playoff_prob'] * team_data['top_pick_prob']
        elif strategy == Strategy.BALANCED:
            win_value = team_data['mean_wins']
            loss_value = 17 - team_data['mean_wins']
            base_value = max(win_value, loss_value)
        
        # Adjust for portfolio synergy/conflicts
        portfolio_adjustment = self._calculate_portfolio_adjustment(
            team, draft_state.my_teams, team_stats
        )
        
        # Adjust for scarcity
        scarcity_adjustment = self._calculate_scarcity_value(
            team, draft_state, team_stats, strategy
        )
        
        return base_value + portfolio_adjustment + scarcity_adjustment
    
    def _calculate_portfolio_adjustment(self,
                                       team: str,
                                       current_teams: List[str],
                                       team_stats: pd.DataFrame) -> float:
        """
        Adjust value based on portfolio fit
        """
        if not current_teams:
            return 0
        
        adjustment = 0
        
        # Penalize if teams play each other (cannibalization)
        # This would need schedule data to be accurate
        # For now, use division as proxy
        
        # Reward diversification
        current_wins = team_stats[team_stats['team'].isin(current_teams)]['mean_wins'].values
        team_wins = team_stats[team_stats['team'] == team]['mean_wins'].values[0]
        
        # Penalize similar win totals (less diversification)
        for curr_wins in current_wins:
            if abs(curr_wins - team_wins) < 2:
                adjustment -= 1
        
        return adjustment
    
    def _calculate_scarcity_value(self,
                                 team: str,
                                 draft_state: DraftState,
                                 team_stats: pd.DataFrame,
                                 strategy: Strategy) -> float:
        """
        Calculate scarcity value - will this team be available later?
        """
        picks_until_next = draft_state.picks_until_next
        if picks_until_next == 0:
            return 0
        
        team_rank = team_stats['mean_wins'].rank(ascending=False)[
            team_stats['team'] == team
        ].values[0]
        
        # Top teams likely to be taken
        if team_rank <= 10:
            scarcity = min(picks_until_next * 0.5, 3)
        elif team_rank >= 23:  # Bottom teams for loss strategy
            scarcity = min(picks_until_next * 0.3, 2)
        else:
            scarcity = 0
        
        return scarcity
    
    def identify_blocking_opportunities(self,
                                       draft_state: DraftState,
                                       team_stats: pd.DataFrame) -> List[Tuple[str, int, float]]:
        """
        Identify teams to draft to block opponents
        
        Returns:
            List of (team, opponent_id, blocking_value)
        """
        blocking_opps = []
        
        for opp_id, opp_teams in draft_state.opponent_teams.items():
            if len(opp_teams) >= 2:
                # Infer opponent strategy
                opp_wins = team_stats[team_stats['team'].isin(opp_teams)]['mean_wins'].mean()
                
                if opp_wins > 9:  # Going for wins
                    # Block good teams
                    targets = team_stats[
                        team_stats['team'].isin(draft_state.available_teams)
                    ].nlargest(3, 'mean_wins')
                    
                    for _, team_data in targets.iterrows():
                        blocking_value = team_data['mean_wins'] * 0.5
                        blocking_opps.append((team_data['team'], opp_id, blocking_value))
                        
                elif opp_wins < 7:  # Going for losses
                    # Block bad teams
                    targets = team_stats[
                        team_stats['team'].isin(draft_state.available_teams)
                    ].nsmallest(3, 'mean_wins')
                    
                    for _, team_data in targets.iterrows():
                        blocking_value = (17 - team_data['mean_wins']) * 0.5
                        blocking_opps.append((team_data['team'], opp_id, blocking_value))
        
        return blocking_opps
    
    def get_recommendations(self,
                          draft_state: DraftState,
                          team_stats: pd.DataFrame,
                          n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get top N team recommendations for current pick
        """
        # Determine strategy
        strategy = self.recommend_strategy(draft_state, team_stats)
        
        # Calculate value for all available teams
        team_values = []
        for team in draft_state.available_teams:
            value = self.calculate_team_value(team, draft_state, team_stats, strategy)
            team_values.append({
                'team': team,
                'value': value,
                'strategy': strategy.value,
                'mean_wins': team_stats[team_stats['team'] == team]['mean_wins'].values[0],
                'mean_losses': 17 - team_stats[team_stats['team'] == team]['mean_wins'].values[0]
            })
        
        # Check for blocking opportunities
        blocking_opps = self.identify_blocking_opportunities(draft_state, team_stats)
        for team, opp_id, block_value in blocking_opps:
            for tv in team_values:
                if tv['team'] == team:
                    tv['value'] += block_value
                    tv['blocking_target'] = opp_id
        
        # Sort by value and return top N
        recommendations = pd.DataFrame(team_values).nlargest(n_recommendations, 'value')
        
        return recommendations


class OpponentModeler:
    """Model opponent strategies based on their picks"""
    
    def __init__(self):
        self.opponent_profiles = {}
    
    def update_opponent_model(self,
                             drafter_id: int,
                             picked_team: str,
                             team_stats: pd.DataFrame):
        """
        Update opponent model based on their pick
        """
        if drafter_id not in self.opponent_profiles:
            self.opponent_profiles[drafter_id] = {
                'teams': [],
                'total_wins': 0,
                'total_losses': 0,
                'strategy': None
            }
        
        profile = self.opponent_profiles[drafter_id]
        profile['teams'].append(picked_team)
        
        team_wins = team_stats[team_stats['team'] == picked_team]['mean_wins'].values[0]
        profile['total_wins'] += team_wins
        profile['total_losses'] += (17 - team_wins)
        
        # Infer strategy
        if len(profile['teams']) >= 2:
            avg_wins = profile['total_wins'] / len(profile['teams'])
            if avg_wins > 9:
                profile['strategy'] = 'WIN_FOCUSED'
            elif avg_wins < 7:
                profile['strategy'] = 'LOSS_FOCUSED'
            else:
                profile['strategy'] = 'BALANCED'
    
    def predict_opponent_targets(self,
                                drafter_id: int,
                                available_teams: List[str],
                                team_stats: pd.DataFrame) -> List[str]:
        """
        Predict which teams an opponent might target
        """
        if drafter_id not in self.opponent_profiles:
            return []
        
        profile = self.opponent_profiles[drafter_id]
        strategy = profile.get('strategy')
        
        if strategy == 'WIN_FOCUSED':
            targets = team_stats[
                team_stats['team'].isin(available_teams)
            ].nlargest(3, 'mean_wins')['team'].tolist()
        elif strategy == 'LOSS_FOCUSED':
            targets = team_stats[
                team_stats['team'].isin(available_teams)
            ].nsmallest(3, 'mean_wins')['team'].tolist()
        else:
            # Balanced - could go either way
            targets = []
        
        return targets
    
    def calculate_threat_level(self, drafter_id: int) -> float:
        """
        Calculate how much of a threat this opponent is
        """
        if drafter_id not in self.opponent_profiles:
            return 0
        
        profile = self.opponent_profiles[drafter_id]
        
        # Threat based on how extreme their portfolio is
        avg_wins = profile['total_wins'] / max(len(profile['teams']), 1)
        
        # More extreme = bigger threat
        threat = abs(avg_wins - 8.5) / 8.5
        
        return threat