data {
    int<lower=1> num_games;
    int<lower=1> num_players;
    int<lower=1> num_players_per_game;
    array[num_games, num_players_per_game] int<lower=1, upper=num_players> home_players;
    array[num_games, num_players_per_game] int<lower=1, upper=num_players> away_players;
    array[num_games, num_players_per_game] int<lower=0> home_players_minutes;
    array[num_games, num_players_per_game] int<lower=0> away_players_minutes;
    array[num_games] real<lower=0, upper=1> results;
}

transformed data {
    real log_90 = log(90);
}

parameters {
    vector[num_players - 1] log_skills_raw;
    real log_home_advantage;
    real<lower=0> kappa;
}

transformed parameters {
    vector[num_players] log_skills = append_row(-sum(log_skills_raw), log_skills_raw);
}

model {
    log_skills_raw ~ normal(0, 1);
    log_home_advantage ~ normal(0, 1);
    kappa ~ normal(0, 1);

    for (game in 1:num_games) {
        vector[num_players_per_game] log_terms_home;
        vector[num_players_per_game] log_terms_away;

        for (p in 1:num_players_per_game) {
            real mins_hp = home_players_minutes[game,p];
            real mins_ap = away_players_minutes[game,p];
            
            if (mins_hp > 0) {
                log_terms_home[p] = log_skills[home_players[game,p]] + log(mins_hp) - log_90;
            } else {
                log_terms_home[p] = negative_infinity();
            }
            
            if (mins_ap > 0) {
                log_terms_away[p] = log_skills[away_players[game,p]] + log(mins_ap) - log_90;
            } else {
                log_terms_away[p] = negative_infinity();
            }
        }

        real log_home_skill = log_sum_exp(log_terms_home);
        real log_away_skill = log_sum_exp(log_terms_away);
        
        real home_strength = exp(log_home_skill + log_home_advantage);
        real away_strength = exp(log_away_skill);
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
        vector[num_players_per_game] log_terms_home;
        vector[num_players_per_game] log_terms_away;

        for (p in 1:num_players_per_game) {
            real mins_hp = home_players_minutes[game,p];
            real mins_ap = away_players_minutes[game,p];
            
            if (mins_hp > 0) {
                log_terms_home[p] = log_skills[home_players[game,p]] + log(mins_hp) - log_90;
            } else {
                log_terms_home[p] = negative_infinity();
            }
            
            if (mins_ap > 0) {
                log_terms_away[p] = log_skills[away_players[game,p]] + log(mins_ap) - log_90;
            } else {
                log_terms_away[p] = negative_infinity();
            }
        }

        real log_home_skill = log_sum_exp(log_terms_home);
        real log_away_skill = log_sum_exp(log_terms_away);
        
        real home_strength = exp(log_home_skill + log_home_advantage);
        real away_strength = exp(log_away_skill);
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
