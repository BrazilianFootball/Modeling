data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0, upper=1> team1_win;
}

parameters {
    sum_to_zero_vector[num_teams] skills_raw;
}

transformed parameters {
    real variance_correction = sqrt(num_teams / (num_teams - 1));
    vector[num_teams] skills = skills_raw - mean(skills_raw);
}

model {
    skills_raw ~ normal(0, 1);

    for (game in 1:num_games) {
        real skill_diff = skills[team1[game]] - skills[team2[game]];
        target += team1_win[game] * skill_diff - log1p_exp(skill_diff);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        real skill_diff = skills[team1[game]] - skills[team2[game]];
        log_lik += team1_win[game] * skill_diff - log1p_exp(skill_diff);
    }
}
