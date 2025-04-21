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
}

transformed parameters {
    vector[num_teams] log_skills;
    log_skills[1:num_teams-1] = log_skills_raw;
    log_skills[num_teams] = -sum(log_skills_raw);
}

model {
    log_skills_raw ~ normal(0, 1);
    
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
