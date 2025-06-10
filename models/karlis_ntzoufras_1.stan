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
    vector[num_teams - 1] raw_ha1;
    vector[num_teams - 1] raw_hd1;
    vector[num_teams - 1] raw_ha2;
    vector[num_teams - 1] raw_hd2;
}

transformed parameters {
    vector[2] h = to_vector([0, raw_h]);
    vector[num_teams] a;
    vector[num_teams] d;
    vector[num_teams] ha1;
    vector[num_teams] hd1;
    vector[num_teams] ha2;
    vector[num_teams] hd2;

    a[1:num_teams - 1] = raw_a;
    a[num_teams] = -sum(raw_a);
    d[1:num_teams - 1] = raw_d;
    d[num_teams] = -sum(raw_d);
    ha1[1] = 0;
    ha1[2:num_teams] = raw_ha1;
    hd1[1] = 0;
    hd1[2:num_teams] = raw_hd1;
    ha2[1:num_teams - 1] = raw_ha2;
    ha2[num_teams] = -sum(raw_ha2);
    hd2[1:num_teams - 1] = raw_hd2;
    hd2[num_teams] = -sum(raw_hd2);
}

model {
    mu ~ normal(0, 1);
    raw_h ~ normal(0, 1);
    raw_a ~ normal(0, 1);
    raw_d ~ normal(0, 1);
    raw_ha1 ~ normal(0, 1);
    raw_hd1 ~ normal(0, 1);
    raw_ha2 ~ normal(0, 1);
    raw_hd2 ~ normal(0, 1);
    for (game in 1:num_games) {
        int j = team1[game];
        int k = team2[game];
        real log_lambda_1jk = mu + h[1] + a[j] + d[k] + ha1[j] + hd2[k];
        real log_lambda_2jk = mu + h[2] + a[k] + d[j] + ha2[k] + hd1[j];
        target += poisson_log_lpmf(goals_team1[game] | log_lambda_1jk);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_2jk);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        int j = team1[game];
        int k = team2[game];
        real log_lambda_1jk = mu + h[1] + a[j] + d[k] + ha1[j] + hd2[k];
        real log_lambda_2jk = mu + h[2] + a[k] + d[j] + ha2[k] + hd1[j];
        
        log_lik += poisson_log_lpmf(goals_team1[game] | log_lambda_1jk);
        log_lik += poisson_log_lpmf(goals_team2[game] | log_lambda_2jk);
    }
}
