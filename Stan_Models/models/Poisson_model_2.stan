data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0> goals_team1;
    array[num_games] int<lower=0> goals_team2;
}

parameters {
    // log skill of each team
    vector<lower=0>[num_teams] skills;
    real<lower=0> home_force;
}

model {
    skills ~ normal(0, 1);
    sum(skills) ~ normal(0, 1e-4);
    real lambda_team1;
    real lambda_team2;
    for (game in 1:num_games) {
        lambda_team1 = exp(skills[team1[game]] + home_force) / exp(skills[team2[game]]);
        lambda_team2 = exp(skills[team2[game]]) / exp(skills[team1[game]] + home_force);
        target += goals_team1[game] * (skills[team1[game]] + home_force - skills[team2[game]]) - lambda_team1;
        target += goals_team2[game] * (skills[team2[game]] - skills[team1[game]] - home_force) - lambda_team2;
    }
}
