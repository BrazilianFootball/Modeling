data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0> goals_team1;
    array[num_games] int<lower=0> goals_team2;
}

parameters {
    real mu;
    real raw_h;
    vector[num_teams - 1] raw_a;
    vector[num_teams - 1] raw_d;
}

transformed parameters {
    vector[2] h = to_vector([0, raw_h]);
    vector[num_teams] a;
    vector[num_teams] d;

    a[1:num_teams - 1] = raw_a;
    a[num_teams] = -sum(raw_a);
    d[1:num_teams - 1] = raw_d;
    d[num_teams] = -sum(raw_d);
}

model {
    mu ~ normal(0, 1);
    raw_h ~ normal(0, 1);
    raw_a ~ normal(0, 1);
    raw_d ~ normal(0, 1);
    for (game in 1:num_games) {
        int j = team1[game];
        int k = team2[game];
        real log_lambda_1jk = mu + h[1] + a[j] + d[k];
        real log_lambda_2jk = mu + h[2] + a[k] + d[j];
        target += poisson_log_lpmf(goals_team1[game] | log_lambda_1jk);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_2jk);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        int j = team1[game];
        int k = team2[game];
        real log_lambda_1jk = mu + h[1] + a[j] + d[k];
        real log_lambda_2jk = mu + h[2] + a[k] + d[j];
        
        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_1jk);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_2jk);
    }
}
