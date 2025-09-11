data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    int<lower=1> num_players_per_club;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team1_players;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team2_players;
    array[num_games] int<lower=0> goals_team1;
    array[num_games] int<lower=0> goals_team2;
}

parameters {
    vector[num_teams * num_players_per_club - 1] raw_alpha;
}

transformed parameters {
    vector[num_teams * num_players_per_club] alpha = append_row(raw_alpha, -sum(raw_alpha));
}

model {
    raw_alpha ~ normal(0, 0.1);

    for (game in 1:num_games) {
        real log_lambda_team1 = sum(alpha[team1_players[game]]) - sum(alpha[team2_players[game]]);
        real log_lambda_team2 = -log_lambda_team1;

        target += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        real log_lambda_team1 = sum(alpha[team1_players[game]]) - sum(alpha[team2_players[game]]);
        real log_lambda_team2 = -log_lambda_team1;

        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}
