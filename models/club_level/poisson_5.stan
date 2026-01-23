data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0> goals_team1;
    array[num_games] int<lower=0> goals_team2;
}

parameters {
    vector[num_teams-1] raw_alpha;
    vector[num_teams-1] raw_gamma;
    vector[num_teams-1] raw_beta;
    vector[num_teams-1] raw_delta;
}

transformed parameters {
    vector[num_teams] alpha = append_row(raw_alpha, -sum(raw_alpha));
    vector[num_teams] gamma = append_row(raw_gamma, -sum(raw_gamma));
    vector[num_teams] beta = append_row(raw_beta, -sum(raw_beta));
    vector[num_teams] delta = append_row(raw_delta, -sum(raw_delta));
}

model {
    raw_alpha ~ normal(0, 1);
    raw_gamma ~ normal(0, 1);
    raw_beta ~ normal(0, 1);
    raw_delta ~ normal(0, 1);

    for (game in 1:num_games) {
        real log_lambda_team1 = alpha[team1[game]] + beta[team2[game]];
        real log_lambda_team2 = gamma[team1[game]] + delta[team2[game]];

        target += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}

generated quantities {
    real log_lik = 0;
    for (game in 1:num_games) {
        real log_lambda_team1 = alpha[team1[game]] + beta[team2[game]];
        real log_lambda_team2 = gamma[team1[game]] + delta[team2[game]];

        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}
