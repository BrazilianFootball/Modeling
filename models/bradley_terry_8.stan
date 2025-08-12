data {
    int<lower=1> num_games;
    int<lower=1> num_teams;
    int<lower=1> num_players_per_club;
    array[num_games] int<lower=1, upper=num_teams> team1;
    array[num_games] int<lower=1, upper=num_teams> team2;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team1_players;
    array[num_games, 11] int<lower=1, upper=num_teams * num_players_per_club> team2_players;
    array[num_games] real<lower=0, upper=1> results;
}

parameters {
    vector[num_teams * num_players_per_club - 1] log_skills_raw;
    real log_home_advantage;
    real<lower=0> kappa;
}

transformed parameters {
    vector[num_teams * num_players_per_club] log_skills = append_row(log_skills_raw, -sum(log_skills_raw));
}

model {
    log_skills_raw ~ normal(0, 1);
    log_home_advantage ~ normal(0, 1);
    kappa ~ normal(0, 1);

    for (game in 1:num_games) {
        real home_strength = exp(sum(log_skills[team1_players[game]]) + log_home_advantage);
        real away_strength = exp(sum(log_skills[team2_players[game]]));
        real tie_strength = kappa * sqrt(home_strength * away_strength);
        real total_strength = home_strength + away_strength + tie_strength;
        
        real prob_home = home_strength / total_strength;
        real prob_away = away_strength / total_strength;
        
        if (results[game] == 1.0) {
            target += log(prob_home);
        } else if (results[game] == 0.0) {
            target += log(prob_away);
        } else {
            target += log(1 - prob_home - prob_away);
        }
    }
}

generated quantities {
    real log_lik = 0;

    for (game in 1:num_games) {
        real home_strength = exp(sum(log_skills[team1_players[game]]) + log_home_advantage);
        real away_strength = exp(sum(log_skills[team2_players[game]]));
        real tie_strength = kappa * sqrt(home_strength * away_strength);
        real total_strength = home_strength + away_strength + tie_strength;
        
        real prob_home = home_strength / total_strength;
        real prob_away = away_strength / total_strength;
        
        if (results[game] == 1.0) {
            log_lik += log(prob_home);
        } else if (results[game] == 0.0) {
            log_lik += log(prob_away);
        } else {
            log_lik += log(1 - prob_home - prob_away);
        }
    }
}
