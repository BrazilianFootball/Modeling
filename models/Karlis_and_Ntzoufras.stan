data {
    int<lower=1> n_games;  // Número total de jogos
    int<lower=1> n_clubs;  // Número total de equipes
    array[n_games] int<lower=1, upper=n_clubs> club_1;  // Equipe 1 em cada jogo
    array[n_games] int<lower=1, upper=n_clubs> club_2;  // Equipe 2 em cada jogo
    array[n_games] int<lower=0> goals_club_1;  // Gols da equipe 1 em cada jogo
    array[n_games] int<lower=0> goals_club_2;  // Gols da equipe 2 em cada jogo
}

parameters {
    real mu;
    vector[2] h;  // Home/away effect
    vector[n_clubs] a; // offensive performance
    vector[n_clubs] d; // defensive performance
    vector[2 * n_clubs] ha; // how offensive performance differs in home and away
    vector[2 * n_clubs] hd; // how defensive performance differs in home and away
}

model {
    h ~ normal(0, 1);
    a ~ normal(0, 1);
    d ~ normal(0, 1);
    mu ~ normal(0, 1);
    ha ~ normal(0, 1);
    hd ~ normal(0, 1);
    for (game in 1:n_games) {
        // Likelihood
        real log_lambda_1jk = mu + h[1] + a[club_1[game]] + d[club_2[game]] + ha[club_1[game]] + hd[n_clubs + club_2[game]];
        real log_lambda_2jk = mu + h[2] + a[club_2[game]] + d[club_1[game]] + ha[n_clubs + club_2[game]] + hd[club_1[game]];
        target += goals_club_1[game] * log_lambda_1jk - exp(log_lambda_1jk);
        target += goals_club_2[game] * log_lambda_2jk - exp(log_lambda_2jk);
    }
}
