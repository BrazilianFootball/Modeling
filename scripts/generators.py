import numpy as np

def create_mask(clubs, force, n_seasons):
    home_teams = np.repeat(clubs, len(clubs))
    away_teams = np.tile(clubs, len(clubs))
    home_force = np.repeat(force, len(clubs))
    away_force = np.tile(force, len(clubs))
    
    home_teams = np.concatenate([home_teams for _ in range(n_seasons)])
    away_teams = np.concatenate([away_teams for _ in range(n_seasons)])
    home_force = np.concatenate([home_force for _ in range(n_seasons)])
    away_force = np.concatenate([away_force for _ in range(n_seasons)])

    mask = home_teams != away_teams
    home_teams = home_teams[mask]
    away_teams = away_teams[mask]
    home_force = home_force[mask]
    away_force = away_force[mask]

    return home_teams, away_teams, home_force, away_force

def data_generator_bt_1(**kwargs):
    np.random.seed(0)
    n_clubs = kwargs.get('n_clubs')
    n_seasons = kwargs.get('n_seasons')
    
    clubs = list(range(1, n_clubs+1))
    force = np.random.normal(size=n_clubs)
    force[-1] = -sum(force[:-1])

    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask
    
    prob_home = np.exp(home_force) / (np.exp(home_force) + np.exp(away_force))
    home_wins = np.zeros_like(prob_home, dtype=int)
    
    random_vals = np.random.uniform(size=n_clubs * (n_clubs - 1) * n_seasons)
    home_wins = (random_vals < prob_home).astype(int)
    
    data = {
        'home_name': home_teams,
        'home_force': home_force,
        'away_name': away_teams,
        'away_force': away_force,
        'home_wins': home_wins
    }
    
    return {
        'variables': {
            'skill': force
        },
        'generated': {
            'num_games': len(home_teams),
            'num_teams': n_clubs,
            'team1': data['home_name'],
            'team2': data['away_name'],
            'team1_win': data['home_wins']
        }
    }


def data_generator_bt_2(**kwargs):
    np.random.seed(0)
    n_clubs = kwargs.get('n_clubs')
    n_seasons = kwargs.get('n_seasons')
    
    clubs = list(range(1, n_clubs+1))
    home_advantage = np.random.normal(0, 1)
    force = np.random.normal(size=n_clubs)
    force[-1] = -sum(force[:-1])
    
    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask
    
    prob_home = np.exp(home_force + home_advantage) / (np.exp(home_force + home_advantage) + np.exp(away_force))
    home_wins = np.zeros_like(prob_home, dtype=int)
    
    random_vals = np.random.uniform(size=n_clubs * (n_clubs - 1) * n_seasons)
    home_wins = (random_vals < prob_home).astype(int)
    
    data = {
        'home_name': home_teams,
        'home_force': home_force,
        'away_name': away_teams,
        'away_force': away_force,
        'home_wins': home_wins
    }
    
    return {
        'variables': {
            'skill': force,
            'home_advantage': home_advantage
        },
        'generated': {
            'num_games': len(home_teams),
            'num_teams': n_clubs,
            'team1': data['home_name'],
            'team2': data['away_name'],
            'team1_win': data['home_wins']
        }
    }

def data_generator_poisson_1(**kwargs):
    np.random.seed(0)
    n_clubs = kwargs.get('n_clubs')
    n_seasons = kwargs.get('n_seasons')
    
    clubs = list(range(1, n_clubs+1))
    
    log_forces = np.random.normal(0, 1, size=n_clubs)
    log_forces[-1] = -sum(log_forces[:-1])
    force = np.exp(log_forces)
    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask
    
    home_goals = np.random.poisson(home_force / away_force)
    away_goals = np.random.poisson(away_force / home_force)
    
    return {
        'variables': {
            'skills': force
        },
        'generated': {
            'num_games': len(home_teams),
            'num_teams': n_clubs,
            'team1': home_teams,
            'team2': away_teams,
            'goals_team1': home_goals,
            'goals_team2': away_goals
        }
    }
