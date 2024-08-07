data {
    int<lower=1> num_games;  // Total number of games
    int<lower=1> num_teams;  // Total number of teams
    array[num_games] int<lower=1, upper=num_teams> team1;  // Team 1 in each game
    array[num_games] int<lower=1, upper=num_teams> team2;  // Team 2 in each game
    array[num_games] int<lower=0, upper=1> team1_win;  // 1 if team1 won, 0 otherwise
}

parameters {
    vector[num_teams] skill;  // Skill of each team
}

model {
    // Normal prior for team skills
    skill ~ normal(0, 1);

    // Soft constraint: sum of skills should be close to 0
    sum(skill) ~ normal(0, 1e-4);

    for (game in 1:num_games) {
        // Likelihood using the Bradley-Terry model
        if (team1_win[game] == 1) {
            target += skill[team1[game]] - log(exp(skill[team1[game]]) + exp(skill[team2[game]]));
        } else {
            target += skill[team2[game]] - log(exp(skill[team1[game]]) + exp(skill[team2[game]]));
        }
    }
}
