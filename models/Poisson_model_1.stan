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
}

model {
    skills[1] ~ normal(1, 1e-4);
    skills[2:num_teams] ~ normal(1, 1);
    real lambda_team1;
    real lambda_team2;
    for (game in 1:num_games) {
        lambda_team1 = skills[team1[game]] / skills[team2[game]];
        lambda_team2 = skills[team2[game]] / skills[team1[game]];
        target += goals_team1[game] * log(lambda_team1) - lambda_team1;
        target += goals_team2[game] * log(lambda_team2) - lambda_team2;
    }
}
