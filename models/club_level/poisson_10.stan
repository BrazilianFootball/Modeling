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
    vector[num_teams] beta;
    vector[num_teams] delta;
    real correlation_strength;
}

transformed parameters {
    vector[num_teams] alpha = append_row(raw_alpha, -sum(raw_alpha));
    vector[num_teams] gamma = append_row(raw_gamma, -sum(raw_gamma));
}

model {
    raw_alpha ~ normal(0, 1);
    raw_gamma ~ normal(0, 1);
    beta ~ normal(0, 1);
    delta ~ normal(0, 1);
    correlation_strength ~ normal(0, 1);

    for (game in 1:num_games) {
        real log_lambda_team1 = alpha[team1[game]] + beta[team2[game]] + correlation_strength;
        real log_lambda_team2 = gamma[team1[game]] + delta[team2[game]] + correlation_strength;

        target += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}

generated quantities {
    real log_lik = 0;
    for (game in 1:num_games) {
        real log_lambda_team1 = alpha[team1[game]] + beta[team2[game]] + correlation_strength;
        real log_lambda_team2 = gamma[team1[game]] + delta[team2[game]] + correlation_strength;

        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}
