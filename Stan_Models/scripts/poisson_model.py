import numpy as np
import pandas as pd
import plotly.express as px

from itertools import product

class championship:
    def __init__(self, DATA, N_SIMS=10_000, DIM_STATS=4) -> None:
        # constants
        self.DATA = DATA
        self.TEAMS_NAME = self.DATA['teams']
        self.HOME_TEAM = np.array(self.DATA['team1'])
        self.AWAY_TEAM = np.array(self.DATA['team2'])
        
        self.N_TEAMS = len(self.TEAMS_NAME)
        self.N_GAMES = self.N_TEAMS * (self.N_TEAMS - 1)
        
        self.N_SIMS = N_SIMS
        self.DIM_STATS = DIM_STATS

        # variables
        self.home_means = np.zeros(self.N_GAMES)
        self.away_means = np.zeros(self.N_GAMES)
        self.simulated_home_scores = np.zeros((self.N_SIMS, self.N_GAMES), dtype=int)
        self.simulated_away_scores = np.zeros((self.N_SIMS, self.N_GAMES), dtype=int)
        self.standings = np.zeros((self.N_SIMS, self.N_TEAMS, self.DIM_STATS), dtype=int)
        self.probabilities = np.zeros((self.N_TEAMS, self.N_TEAMS))

    def update_parameters(self, initial_round, mode='random'):
        home_goals = np.array(self.DATA['goals_team1'][:(initial_round - 1) * 10])
        away_goals = np.array(self.DATA['goals_team2'][:(initial_round - 1) * 10])
        home_team = np.array(self.DATA['team1'][:(initial_round - 1) * 10])
        away_team = np.array(self.DATA['team2'][:(initial_round - 1) * 10])
        complete_home_team = np.array(self.DATA['team1'])
        complete_away_team = np.array(self.DATA['team2'])

        self.home_means = np.zeros(self.N_GAMES)
        self.away_means = np.zeros(self.N_GAMES)
        if mode == 'random' or initial_round < 5:
            self.home_means += np.abs(np.random.normal(size=self.N_GAMES))
            self.away_means += np.abs(np.random.normal(size=self.N_GAMES))
        elif mode == 'simple_poisson':
            lam = np.mean(np.hstack((home_goals, away_goals)))
            self.home_means += lam * np.ones(self.N_GAMES)
            self.away_means += lam * np.ones(self.N_GAMES)
        elif mode == 'club_poisson':
            for team in range(1, self.N_TEAMS + 1):
                goals = np.hstack((home_goals[home_team == team], away_goals[away_team == team]))
                lam = np.mean(goals)
                self.home_means += lam * (complete_home_team == team)
                self.away_means += lam * (complete_away_team == team)
        elif mode == 'club_poisson_v2':
            for team in range(1, self.N_TEAMS + 1):
                home_lam = np.mean(home_goals[home_team == team])
                away_lam = np.mean(away_goals[away_team == team])
                self.home_means += home_lam * (complete_home_team == team)
                self.away_means += away_lam * (complete_away_team == team)
        else:
            df = pd.read_csv(mode.format(actual_round=initial_round))
            for i in range(1, self.N_TEAMS + 1):
                self.home_means[self.HOME_TEAM == i] = df.loc[i, 'mean']
                self.away_means[self.AWAY_TEAM == i] = df.loc[i, 'mean']
            
            if 'home_force' in df['variable'].values:
                home_force_inx = np.argmax('home_force' == df['variable'].values)
                self.home_means += df.loc[home_force_inx, 'mean']
            
            home_save = self.home_means
            self.home_means = self.home_means / self.away_means
            self.away_means = self.away_means / home_save

    def simulate_games(self, initial_round=1):
        if self.home_means.shape == (self.N_SIMS, self.N_GAMES):
            self.simulated_home_scores = np.random.poisson(self.home_means)
            self.simulated_away_scores = np.random.poisson(self.away_means)
        else:
            self.simulated_home_scores = np.random.poisson(self.home_means, size=(self.N_SIMS, self.N_GAMES))
            self.simulated_away_scores = np.random.poisson(self.away_means, size=(self.N_SIMS, self.N_GAMES))

        if initial_round > 1:
            observed_home_goals = np.array(self.DATA['goals_team1'][:(initial_round - 1) * 10])
            observed_away_goals = np.array(self.DATA['goals_team2'][:(initial_round - 1) * 10])
            self.simulated_home_scores[:, :(initial_round - 1) * 10] = observed_home_goals
            self.simulated_away_scores[:, :(initial_round - 1) * 10] = observed_away_goals
    
    def calculate_standings(self):
        self.standings = np.zeros((self.N_SIMS, self.N_TEAMS, self.DIM_STATS), dtype=int)
        for i in range(self.N_TEAMS):
            # points
            self.standings[:, i, 0] += 3 * np.sum(
                self.simulated_home_scores[:, self.HOME_TEAM == i + 1] > self.simulated_away_scores[:, self.HOME_TEAM == i + 1], axis=1
            )
            self.standings[:, i, 0] += 3 * np.sum(
                self.simulated_away_scores[:, self.AWAY_TEAM == i + 1] > self.simulated_home_scores[:, self.AWAY_TEAM == i + 1], axis=1
            )
            self.standings[:, i, 0] += 1 * np.sum(
                self.simulated_home_scores[:, self.HOME_TEAM == i + 1] == self.simulated_away_scores[:, self.HOME_TEAM == i + 1], axis=1
            )
            self.standings[:, i, 0] += 1 * np.sum(
                self.simulated_away_scores[:, self.AWAY_TEAM == i + 1] == self.simulated_home_scores[:, self.AWAY_TEAM == i + 1], axis=1
            )

            # wins
            self.standings[:, i, 1] += 1 * np.sum(
                self.simulated_home_scores[:, self.HOME_TEAM == i + 1] > self.simulated_away_scores[:, self.HOME_TEAM == i + 1], axis=1
            )
            self.standings[:, i, 1] += 1 * np.sum(
                self.simulated_away_scores[:, self.AWAY_TEAM == i + 1] > self.simulated_home_scores[:, self.AWAY_TEAM == i + 1], axis=1
            )

            # goals for
            self.standings[:, i, 2] += np.sum(self.simulated_home_scores[:, self.HOME_TEAM == i + 1], axis=1)
            self.standings[:, i, 2] += np.sum(self.simulated_away_scores[:, self.AWAY_TEAM == i + 1], axis=1)

            # goals difference
            self.standings[:, i, 3] += np.sum(
                self.simulated_home_scores[:, self.HOME_TEAM == i + 1] - self.simulated_away_scores[:, self.HOME_TEAM == i + 1], axis=1
            )
            self.standings[:, i, 3] += np.sum(
                self.simulated_away_scores[:, self.AWAY_TEAM == i + 1] - self.simulated_home_scores[:, self.AWAY_TEAM == i + 1], axis=1
            )
    
    def generate_final_position(self, column_indices=[0, 1, 2, 3]):
        indices = np.argsort(self.standings[..., column_indices[-1]], axis=-1)
        for col_idx in column_indices[-2::-1]:
            indices = np.take_along_axis(
                indices,
                np.argsort(
                    np.take_along_axis(
                        self.standings[..., col_idx],
                        indices,
                        axis=-1
                    ),
                    axis=-1
                ),
                axis=-1
            )
        
        self.standings = indices[..., ::-1]
    
    def calculate_final_position_probabilities(self):
        for club, position in product(range(self.N_TEAMS), repeat=2):
            self.probabilities[club, position] += sum(self.standings[:, position] == club)

        self.probabilities = self.probabilities / self.N_SIMS
    
    def simulate_championship(self, initial_round, mode):
        self.update_parameters(initial_round, mode)
        self.simulate_games(initial_round)
        self.calculate_standings()
        self.generate_final_position()
        self.calculate_final_position_probabilities()
    
    def run_simulations(self, initial_rounds, mode_list, club_list):
        probs = dict()
        for initial_round, mode in product(initial_rounds, mode_list):
            mode_name = mode[1]
            if initial_round not in probs: probs[initial_round] = dict()
            if mode_name not in probs[initial_round]: probs[initial_round][mode_name] = dict()
            self.simulate_championship(initial_round=initial_round, mode=mode[0])
            for i in range(self.N_TEAMS):
                team = self.TEAMS_NAME[i]
                if team in club_list:
                    probs[initial_round][mode_name][team] = self.probabilities[i][0]
        
        return probs
    
    def generate_plot(self, initial_rounds, mode_list, club_list, probs):
        data = []
        for initial_round in initial_rounds:
            for mode in mode_list:
                mode_name = mode[1]
                for team in club_list:
                    data.append({
                        'Initial Round': initial_round,
                        'Mode': mode_name,
                        'Team': team,
                        'Probability': probs[initial_round][mode_name][team]
                    })

        df = pd.DataFrame(data)

        fig = px.line(
            df,
            x='Initial Round',
            y='Probability',
            color='Team',
            line_dash='Mode' if len(mode_list) > 1 else None,
            title='Evolution of Each Team\'s Probability Over Rounds and Modes' if len(mode_list) > 1 else 'Evolution of Each Team\'s Probability Over Rounds',
            labels={
                'Initial Round': 'Initial Round',
                'Probability': 'Probability',
                'Team': 'Team',
                'Mode': 'Mode'
            }
        )

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                tickmode='array',
                tickvals=[*initial_rounds],
                showline=True,
                linewidth=2,
                linecolor='black',
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                gridcolor='lightgray'
            ),
            legend_title_text='Teams and Modes' if len(mode_list) > 1 else 'Teams'
        )

        fig.show('png', width=1200)
