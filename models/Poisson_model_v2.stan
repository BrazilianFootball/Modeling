data {
    int<lower=1> n_games;
    int<lower=1> n_teams;
    int<lower=1> n_players;
    int<lower=1> total_time_per_game;
    array[n_games] int<lower=0, upper=n_teams> home_team;
    array[n_games] int<lower=0, upper=n_teams> away_team;
    array[n_games] int<lower=0> home_score;
    array[n_games] int<lower=0> away_score;
    array[n_games, 11] int<lower=1> home_players;
    array[n_games, 11] int<lower=1> away_players;
    array[n_games] int<lower=1> time_played;
}

parameters {
    vector<lower=0>[n_players] skills;
}

model {
    skills[1] ~ normal(1, 1e-4);
    skills[2:n_players] ~ normal(1, 1);
    real lambda_home;
    real lambda_away;
    real home_skill;
    real away_skill;
    for (game in 1:n_games) {
        home_skill = sum(skills[home_players[game]]) * time_played[game] / total_time_per_game;
        away_skill = sum(skills[away_players[game]]) * time_played[game] / total_time_per_game;
        lambda_home = home_skill / away_skill;
        lambda_away = away_skill / home_skill;

        target += home_score[game] * log(lambda_home) - lambda_home;
        target += away_score[game] * log(lambda_away) - lambda_away;
    }
}