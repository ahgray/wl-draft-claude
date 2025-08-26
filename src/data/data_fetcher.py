"""
NFL Data Fetcher - Consolidates data from multiple sources
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup
import pickle

class NFLDataFetcher:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "nfl_data.db"
        self.teams = self._initialize_teams()
        
    def _initialize_teams(self) -> Dict:
        """Initialize NFL team abbreviations and names"""
        teams = {
            'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons',
            'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
            'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
            'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
            'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos',
            'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
            'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts',
            'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
            'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
            'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
            'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots',
            'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
            'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles',
            'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
            'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers',
            'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
        }
        return teams
    
    def load_massey_ratings(self, filepath: str = "massey-2025-export.csv") -> pd.DataFrame:
        """Load and parse Massey ratings from CSV"""
        try:
            df = pd.read_csv(filepath)
            
            # Clean up the data
            df['Team_Clean'] = df['Team'].str.strip()
            df['Rating'] = pd.to_numeric(df['Rat'], errors='coerce')
            df['PowerRank'] = pd.to_numeric(df['Pwr'], errors='coerce')
            df['OffenseRating'] = pd.to_numeric(df['Off'], errors='coerce')
            df['DefenseRating'] = pd.to_numeric(df['Def'], errors='coerce')
            df['ExpectedWins'] = pd.to_numeric(df['EW'], errors='coerce')
            df['ExpectedLosses'] = pd.to_numeric(df['EL'], errors='coerce')
            
            # Map team names to abbreviations
            team_mapping = {
                'Baltimore': 'BAL', 'Philadelphia': 'PHI', 'Buffalo': 'BUF',
                'Kansas City': 'KC', 'Green Bay': 'GB', 'Detroit': 'DET',
                'LA Rams': 'LAR', 'Tampa Bay': 'TB', 'San Francisco': 'SF',
                'Minnesota': 'MIN', 'Cincinnati': 'CIN', 'LA Chargers': 'LAC',
                'Denver': 'DEN', 'Pittsburgh': 'PIT', 'Dallas': 'DAL',
                'Seattle': 'SEA', 'Washington': 'WAS', 'Miami': 'MIA',
                'Arizona': 'ARI', 'Houston': 'HOU', 'Chicago': 'CHI',
                'New Orleans': 'NO', 'Atlanta': 'ATL', 'Las Vegas': 'LV',
                'NY Jets': 'NYJ', 'Cleveland': 'CLE', 'Indianapolis': 'IND',
                'New England': 'NE', 'NY Giants': 'NYG', 'Jacksonville': 'JAX',
                'Tennessee': 'TEN', 'Carolina': 'CAR'
            }
            
            df['team_abbr'] = df['Team_Clean'].map(team_mapping)
            
            # Select relevant columns
            ratings_df = df[['team_abbr', 'Rating', 'PowerRank', 'OffenseRating', 
                           'DefenseRating', 'ExpectedWins', 'ExpectedLosses']].copy()
            
            print(f"Loaded Massey ratings for {len(ratings_df)} teams")
            return ratings_df
            
        except Exception as e:
            print(f"Error loading Massey ratings: {e}")
            return pd.DataFrame()
    
    def fetch_nfl_schedule(self, year: int = 2025) -> pd.DataFrame:
        """Fetch NFL schedule using nfl_data_py"""
        try:
            import nfl_data_py as nfl
            
            # Fetch schedule
            schedule = nfl.import_schedules([year])
            
            # Clean and prepare schedule data
            schedule_clean = schedule[['game_id', 'season', 'week', 'gameday', 
                                      'home_team', 'away_team', 'div_game']].copy()
            schedule_clean['is_divisional'] = schedule_clean['div_game'] == 1
            
            print(f"Loaded {len(schedule_clean)} games for {year} season")
            return schedule_clean
            
        except Exception as e:
            print(f"Error fetching NFL schedule: {e}")
            return pd.DataFrame()
    
    def fetch_vegas_odds(self) -> Dict:
        """Fetch Vegas win totals (would need actual API or scraping)"""
        # Placeholder - in reality would scrape from ESPN or odds site
        vegas_totals = {
            'BAL': 10.5, 'PHI': 10.5, 'BUF': 10.5, 'KC': 10.5, 'GB': 10.0,
            'DET': 9.5, 'LAR': 9.5, 'TB': 9.5, 'SF': 9.5, 'MIN': 9.0,
            'CIN': 9.0, 'LAC': 8.5, 'DEN': 8.5, 'PIT': 8.5, 'DAL': 8.5,
            'SEA': 8.5, 'WAS': 8.0, 'MIA': 8.0, 'ARI': 8.0, 'HOU': 8.0,
            'CHI': 7.5, 'NO': 7.5, 'ATL': 7.5, 'LV': 7.0, 'NYJ': 7.0,
            'CLE': 6.5, 'IND': 6.5, 'NE': 6.5, 'NYG': 6.5, 'JAX': 6.5,
            'TEN': 6.0, 'CAR': 5.5
        }
        return vegas_totals
    
    def create_composite_rankings(self) -> pd.DataFrame:
        """Combine all data sources into composite rankings"""
        
        # Load Massey ratings
        massey = self.load_massey_ratings()
        
        # Get Vegas totals
        vegas = self.fetch_vegas_odds()
        vegas_df = pd.DataFrame(list(vegas.items()), columns=['team_abbr', 'vegas_wins'])
        
        # Merge data sources
        composite = massey.merge(vegas_df, on='team_abbr', how='outer')
        
        # Create composite rating (weighted average)
        composite['composite_rating'] = (
            composite['Rating'] * 0.4 +  # 40% Massey
            composite['vegas_wins'] * 0.3 +  # 30% Vegas
            composite['ExpectedWins'] * 0.3  # 30% Massey expected wins
        )
        
        # Rank teams
        composite['win_rank'] = composite['composite_rating'].rank(ascending=False)
        composite['loss_rank'] = composite['composite_rating'].rank(ascending=True)
        
        # Add variance estimates for simulation
        composite['rating_std'] = 2.5  # Standard deviation for simulation
        
        print(f"Created composite rankings for {len(composite)} teams")
        return composite
    
    def save_to_database(self, df: pd.DataFrame, table_name: str):
        """Save dataframe to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Saved {len(df)} rows to {table_name}")
    
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        """Load dataframe from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    
    def cache_all_data(self) -> Dict:
        """Fetch and cache all data for draft day"""
        print("Fetching and caching all data...")
        
        # Get all data
        rankings = self.create_composite_rankings()
        schedule = self.fetch_nfl_schedule()
        
        # Save to database
        self.save_to_database(rankings, 'team_rankings')
        self.save_to_database(schedule, 'schedule_2025')
        
        # Also save as pickle for fast loading
        data = {
            'rankings': rankings,
            'schedule': schedule,
            'timestamp': datetime.now(),
            'teams': self.teams
        }
        
        with open(self.cache_dir / 'all_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print("All data cached successfully!")
        return data
    
    def load_cached_data(self) -> Dict:
        """Load all cached data"""
        cache_file = self.cache_dir / 'all_data.pkl'
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded cached data from {data['timestamp']}")
            return data
        else:
            print("No cached data found. Running cache_all_data()...")
            return self.cache_all_data()


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = NFLDataFetcher()
    data = fetcher.cache_all_data()
    print("\nTop 5 teams by composite rating:")
    print(data['rankings'].nlargest(5, 'composite_rating')[['team_abbr', 'composite_rating', 'ExpectedWins']])
    print("\nBottom 5 teams by composite rating:")
    print(data['rankings'].nsmallest(5, 'composite_rating')[['team_abbr', 'composite_rating', 'ExpectedWins']])