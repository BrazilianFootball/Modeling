data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0> goals_team1;
    array[num_games] int<lower=0> goals_team2;
}

parameters {
    vector[num_teams-1] log_skills_raw;
    real<lower=0> home_force;
}

transformed parameters {
    vector[num_teams] log_skills;
    log_skills[1:num_teams-1] = log_skills_raw;
    log_skills[num_teams] = -sum(log_skills_raw);
}

model {
    log_skills_raw ~ normal(0, 1);
    home_force ~ normal(1, 1);
    real aux = log(home_force);
    for (game in 1:num_games) {
        real log_lambda_team1 = log_sum_exp(log_skills[team1[game]], aux) - log_skills[team2[game]];
        real log_lambda_team2 = -log_lambda_team1;

        target += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}

generated quantities {
    real log_lik = 0;
    real aux = log(home_force);
    for (game in 1:num_games) {
        real log_lambda_team1 = log_sum_exp(log_skills[team1[game]], aux) - log_skills[team2[game]];
        real log_lambda_team2 = -log_lambda_team1;

        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}
