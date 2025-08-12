data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    int<lower=1> num_players_per_club;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team1_players;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team2_players;
    array[num_games] int<lower=0> goals_team1;
    array[num_games] int<lower=0> goals_team2;
}

parameters {
    vector[num_teams * num_players_per_club - 1] raw_alpha;
    vector[num_teams * num_players_per_club - 1] raw_gamma;
    vector[num_teams * num_players_per_club] beta;
    vector[num_teams * num_players_per_club] delta;
}

transformed parameters {
    vector[num_teams * num_players_per_club] alpha = append_row(raw_alpha, -sum(raw_alpha));
    vector[num_teams * num_players_per_club] gamma = append_row(raw_gamma, -sum(raw_gamma));
}

model {
    raw_alpha ~ normal(0, 1);
    raw_gamma ~ normal(0, 1);
    beta ~ normal(0, 1);
    delta ~ normal(0, 1);

    for (game in 1:num_games) {
        real log_lambda_team1 = sum(alpha[team1_players[game]]) + sum(beta[team2_players[game]]);
        real log_lambda_team2 = sum(gamma[team1_players[game]]) + sum(delta[team2_players[game]]);

        target += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}

generated quantities {
    real log_lik = 0;
    for (game in 1:num_games) {
        real log_lambda_team1 = sum(alpha[team1_players[game]]) + sum(beta[team2_players[game]]);
        real log_lambda_team2 = sum(gamma[team1_players[game]]) + sum(delta[team2_players[game]]);

        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}
