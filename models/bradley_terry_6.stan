data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    int<lower=1> num_players_per_club;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team1_players;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team2_players;
    array[num_games] int<lower=0, upper=1> results;
}

parameters {
    vector[num_teams * num_players_per_club - 1] log_skills_raw;
    real log_home_advantage;
}

transformed parameters {
    vector[num_teams * num_players_per_club] log_skills = append_row(log_skills_raw, -sum(log_skills_raw));
}

model {
    log_skills_raw ~ normal(0, 1);
    log_home_advantage ~ normal(0, 1);

    for (game in 1:num_games) {
        real log_force_team1 = sum(log_skills[team1_players[game]]) + log_home_advantage;
        real log_force_team2 = sum(log_skills[team2_players[game]]);
        real log_skill_diff = log_force_team1 - log_force_team2;
        target += results[game] * log_skill_diff - log1p_exp(log_skill_diff);
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        real log_force_team1 = sum(log_skills[team1_players[game]]) + log_home_advantage;
        real log_force_team2 = sum(log_skills[team2_players[game]]);
        real log_skill_diff = log_force_team1 - log_force_team2;
        log_lik += results[game] * log_skill_diff - log1p_exp(log_skill_diff);
    }
}
