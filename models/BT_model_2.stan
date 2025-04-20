data {
    int<lower=1> num_games;  // Total number of games
    int<lower=1> num_teams;  // Total number of teams
    array[num_games] int<lower=1, upper=num_teams> team1;  // Team 1 in each game
    array[num_games] int<lower=1, upper=num_teams> team2;  // Team 2 in each game
    array[num_games] int<lower=0, upper=1> team1_win;  // 1 if team1 won, 0 otherwise
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
    // Normal prior for team skills
    skill_raw ~ normal(0, 1);

    for (game in 1:num_games) {
        // Likelihood using the Bradley-Terry model
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
