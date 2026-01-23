data {
    int<lower=1> num_games;
    int<lower=1> num_players;
    int<lower=1> num_players_per_game;
    array[num_games, num_players_per_game] int<lower=1, upper=num_players> home_players;
    array[num_games, num_players_per_game] int<lower=1, upper=num_players> away_players;
    array[num_games, num_players_per_game] int<lower=0> home_players_minutes;
    array[num_games, num_players_per_game] int<lower=0> away_players_minutes;
    array[num_games] int<lower=0> home_goals;
    array[num_games] int<lower=0> away_goals;
}

transformed data {
    real log_90 = log(90);
}

parameters {
    vector[num_players - 1] raw_alpha;
    vector[num_players - 1] raw_beta;
}

transformed parameters {
    vector[num_players] alpha = append_row(-sum(raw_alpha), raw_alpha);
    vector[num_players] beta = append_row(-sum(raw_beta), raw_beta);
}

model {
    raw_alpha ~ normal(0, 0.1);
    raw_beta ~ normal(0, 0.1);

    for (game in 1:num_games) {
        vector[num_players_per_game] log_terms_home_attack;
        vector[num_players_per_game] log_terms_away_defense;
        vector[num_players_per_game] log_terms_away_attack;
        vector[num_players_per_game] log_terms_home_defense;

        for (p in 1:num_players_per_game) {
            real mins_hp = home_players_minutes[game,p];
            real mins_ap = away_players_minutes[game,p];
            
            if (mins_hp > 0) {
                log_terms_home_attack[p] = alpha[home_players[game,p]] + log(mins_hp) - log_90;
                log_terms_home_defense[p] = beta[home_players[game,p]] + log(mins_hp) - log_90;
            } else {
                log_terms_home_attack[p] = negative_infinity();
                log_terms_home_defense[p] = negative_infinity();
            }
            
            if (mins_ap > 0) {
                log_terms_away_defense[p] = beta[away_players[game,p]] + log(mins_ap) - log_90;
                log_terms_away_attack[p] = alpha[away_players[game,p]] + log(mins_ap) - log_90;
            } else {
                log_terms_away_defense[p] = negative_infinity();
                log_terms_away_attack[p] = negative_infinity();
            }
        }

        real log_home_attack = log_sum_exp(log_terms_home_attack);
        real log_away_defense = log_sum_exp(log_terms_away_defense);
        real log_away_attack = log_sum_exp(log_terms_away_attack);
        real log_home_defense = log_sum_exp(log_terms_home_defense);

        real log_lambda_h = log_home_attack + log_away_defense;
        real log_lambda_a = log_away_attack + log_home_defense;
        
        target += poisson_log_lpmf(home_goals[game] | log_lambda_h);
        target += poisson_log_lpmf(away_goals[game] | log_lambda_a);
    }
}

generated quantities {
    real log_lik = 0;
    for (game in 1:num_games) {
        vector[num_players_per_game] log_terms_home_attack;
        vector[num_players_per_game] log_terms_away_defense;
        vector[num_players_per_game] log_terms_away_attack;
        vector[num_players_per_game] log_terms_home_defense;

        for (p in 1:num_players_per_game) {
            real mins_hp = home_players_minutes[game,p];
            real mins_ap = away_players_minutes[game,p];
            
            if (mins_hp > 0) {
                log_terms_home_attack[p] = alpha[home_players[game,p]] + log(mins_hp) - log_90;
                log_terms_home_defense[p] = beta[home_players[game,p]] + log(mins_hp) - log_90;
            } else {
                log_terms_home_attack[p] = negative_infinity();
                log_terms_home_defense[p] = negative_infinity();
            }
            
            if (mins_ap > 0) {
                log_terms_away_defense[p] = beta[away_players[game,p]] + log(mins_ap) - log_90;
                log_terms_away_attack[p] = alpha[away_players[game,p]] + log(mins_ap) - log_90;
            } else {
                log_terms_away_defense[p] = negative_infinity();
                log_terms_away_attack[p] = negative_infinity();
            }
        }

        real log_home_attack = log_sum_exp(log_terms_home_attack);
        real log_away_defense = log_sum_exp(log_terms_away_defense);
        real log_away_attack = log_sum_exp(log_terms_away_attack);
        real log_home_defense = log_sum_exp(log_terms_home_defense);

        real log_lambda_h = log_home_attack + log_away_defense;
        real log_lambda_a = log_away_attack + log_home_defense;
        
        log_lik += poisson_log_lpmf(home_goals[game] | log_lambda_h);
        log_lik += poisson_log_lpmf(away_goals[game] | log_lambda_a);
    }
}
