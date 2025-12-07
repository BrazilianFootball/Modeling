data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games] int<lower=0, upper=1> results;
}

parameters {
    vector[num_teams-1] log_skills_raw;
}

transformed parameters {
    vector[num_teams] log_skills = append_row(log_skills_raw, -sum(log_skills_raw));
}

model {
    log_skills_raw ~ normal(0, 1);

    for (game in 1:num_games) {
        real log_skill_diff = log_skills[team1[game]] - log_skills[team2[game]];
        target += results[game] * log_skill_diff - log1p_exp(log_skill_diff);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        real log_skill_diff = log_skills[team1[game]] - log_skills[team2[game]];
        log_lik += results[game] * log_skill_diff - log1p_exp(log_skill_diff);
    }
}
