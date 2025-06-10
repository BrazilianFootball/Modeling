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
    vector[2] h;  // Home/away effect
    vector[num_teams] a; // offensive performance
    vector[num_teams] d; // defensive performance
    vector[2 * num_teams] ha; // how offensive performance differs in home and away
    vector[2 * num_teams] hd; // how defensive performance differs in home and away
}

model {
    h ~ normal(0, 1);
    a ~ normal(0, 1);
    d ~ normal(0, 1);
    mu ~ normal(0, 1);
    ha ~ normal(0, 1);
    hd ~ normal(0, 1);
    for (game in 1:num_games) {
        // Likelihood
        int j = team1[game];
        int k = team2[game];
        real log_lambda_1jk = mu + h[1] + a[j] + d[k] + ha[j] + hd[num_teams + k];
        real log_lambda_2jk = mu + h[2] + a[k] + d[j] + ha[num_teams + k] + hd[j];
        target += poisson_log_lpmf(goals_team1[game] | log_lambda_1jk);
        target += poisson_log_lpmf(goals_team2[game] | log_lambda_2jk);
    }
}
