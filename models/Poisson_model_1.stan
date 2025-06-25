data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0> goals_team1;
    array[num_games] int<lower=0> goals_team2;
}

parameters {
    sum_to_zero_vector[num_teams] log_skills;
}

transformed parameters {
    real variance_correction = sqrt(num_teams / (num_teams - 1));
}

model {
    log_skills ~ normal(0, variance_correction * 1);
    
    for (game in 1:num_games) {
        real log_lambda_team1 = log_skills[team1[game]] - log_skills[team2[game]];
        real log_lambda_team2 = -log_lambda_team1;
        
        target += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        real log_lambda_team1 = log_skills[team1[game]] - log_skills[team2[game]];
        real log_lambda_team2 = -log_lambda_team1;

        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_team1);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_team2);
    }
}
