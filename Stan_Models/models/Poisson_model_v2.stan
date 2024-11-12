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
    vector<lower=0>[n_players+1] skills;
}

model {
    skills ~ normal(0, 1);
    skills[n_players+1] ~ normal(0, 1e-8);
    for (game in 1:n_games) {
        real home_skill = sum(skills[home_players[game]]) * time_played[game] / total_time_per_game;
        real away_skill = sum(skills[away_players[game]]) * time_played[game] / total_time_per_game;

        // Likelihood
        target += home_score[game] * (log(home_skill) - log(away_skill)) - (home_skill / away_skill);
        target += away_score[game] * (log(away_skill) - log(home_skill)) - (away_skill / home_skill);
    }
}