data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0, upper=1> team1_win;
}

parameters {
    vector[num_teams-1] skill_raw;
    real home_advantage;
}

transformed parameters {
    vector[num_teams] skill;
    skill[1:num_teams-1] = skill_raw;
    skill[num_teams] = -sum(skill_raw);
}

model {
    skill_raw ~ normal(0, 1);

    for (game in 1:num_games) {
        real skill_diff = skill[team1[game]] + home_advantage - skill[team2[game]];
        target += team1_win[game] * skill_diff - log1p_exp(skill_diff);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        real skill_diff = skill[team1[game]] + home_advantage - skill[team2[game]];
        log_lik += team1_win[game] * skill_diff - log1p_exp(skill_diff);
    }
}
