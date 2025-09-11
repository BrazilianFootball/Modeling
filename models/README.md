# Implemented Models

## Bradley-Terry Models

### `bt_model_1.stan`
- **Description**: Bradley-Terry model without home advantage
- **Observed Data**:
  - $N$: total number of pairwise comparisons (games)
  - $K$: total number of teams in the competition
  - Team matchups: pairs of teams $(i, j)$ that competed against each other
  - Game outcomes: binary results indicating whether the first team won each comparison

- **Parameters:**
  - **Team Skills**: Each team has an underlying skill level $\theta_i$ that represents their competitive strength relative to other teams.
  - **Identifiability Constraint**: To ensure the model is identifiable, we impose a sum-to-zero constraint on the team skills, meaning they are measured relative to the average team strength.

  **Mathematical Formulation:**
  $$\theta_i \sim \mathcal{N}(0, 1) \quad \text{for } i = 1, \ldots, K-1$$
  $$\theta_K = -\sum_{j=1}^{K-1} \theta_j$$

  This ensures that $\sum_{i=1}^{K} \theta_i = 0$, preventing the model from having infinitely many equivalent solutions.

- **Likelihood:**

  For each game $g$ between team $i$ and team $j$:
  $$\delta_{i,j} = \theta_i - \theta_j$$
  $$P(\text{team } i \text{ wins}) = \frac{1}{1 + \exp(-\delta_{i,j})} = \text{logit}^{-1}(\delta_{i,j})$$

  The game outcome follows a Bernoulli distribution:
  $$w_g \sim \text{Bernoulli}\left(\frac{1}{1 + \exp(-(\theta_{i_g} - \theta_{j_g}))}\right)$$

  where $w_g = 1$ if team $i$ wins and $w_g = 0$ if team $j$ wins.

  The log-likelihood contribution for each game is:
  $$\log p(w_g | \theta) = w_g \cdot \delta_{i,j} - \log(1 + \exp(\delta_{i,j}))$$

### `bt_model_2.stan`
- **Description**: Bradley-Terry model with home advantage
- **Observed Data**:
  - $N$: total number of pairwise comparisons (games)
  - $K$: total number of teams in the competition
  - Team matchups: pairs of teams $(i, j)$ that competed against each other
  - Game outcomes: binary results indicating whether the first team won each comparison

- **Parameters:**
  - **Team Skills**: Each team has an underlying skill level $\theta_i$ that represents their competitive strength relative to other teams.
  - **Home Advantage**: A parameter $\alpha$ that represents the additional advantage gained by playing at home.
  - **Identifiability Constraint**: To ensure the model is identifiable, we impose a sum-to-zero constraint on the team skills, meaning they are measured relative to the average team strength.

  **Mathematical Formulation:**
  $$\theta_i \sim \mathcal{N}(0, 1) \quad \text{for } i = 1, \ldots, K-1$$
  $$\theta_K = -\sum_{j=1}^{K-1} \theta_j$$
  $$\alpha \sim \mathcal{N}(0, 1)$$

  This ensures that $\sum_{i=1}^{K} \theta_i = 0$, preventing the model from having infinitely many equivalent solutions.

- **Likelihood:**

  For each game $g$ between team $i$ (home) and team $j$ (away):
  $$\delta_{i,j} = \theta_i + \alpha - \theta_j$$
  $$P(\text{team } i \text{ wins at home}) = \frac{1}{1 + \exp(-\delta_{i,j})} = \text{logit}^{-1}(\delta_{i,j})$$

  The game outcome follows a Bernoulli distribution:
  $$w_g \sim \text{Bernoulli}\left(\frac{1}{1 + \exp(-((\theta_{i_g} + \alpha) - \theta_{j_g}))}\right)$$

  where $w_g = 1$ if the home team $i$ wins and $w_g = 0$ if the away team $j$ wins.

  The log-likelihood contribution for each game is:
  $$\log p(w_g | \theta, \alpha) = w_g \cdot \delta_{i,j} - \log(1 + \exp(\delta_{i,j}))$$

## Poisson Models

### `poisson_model_1.stan`
- **Description**: Poisson model without home advantage for goal scoring
- **Observed Data**:
  - $N$: total number of games
  - $K$: total number of teams in the competition
  - Team matchups: pairs of teams $(i, j)$ that competed against each other
  - Goal counts: number of goals scored by each team in each game

- **Parameters:**
  - **Team Log-Skills**: Each team has an underlying log-skill level $\phi_i$ that represents their goal-scoring ability relative to other teams.
  - **Identifiability Constraint**: To ensure the model is identifiable, we impose a sum-to-zero constraint on the log-skills, meaning they are measured relative to the average team strength.

  **Mathematical Formulation:**
  $$\phi_i \sim \mathcal{N}(0, 1) \quad \text{for } i = 1, \ldots, K-1$$
  $$\phi_K = -\sum_{j=1}^{K-1} \phi_j$$

  This ensures that $\sum_{i=1}^{K} \phi_i = 0$, preventing the model from having infinitely many equivalent solutions.

- **Likelihood:**

  For each game $g$ between team $i$ and team $j$:
  $$\lambda_{i,j} = \exp(\phi_i - \phi_j)$$
  $$\lambda_{j,i} = \exp(\phi_j - \phi_i) = \frac{1}{\lambda_{i,j}}$$

  The goal counts follow Poisson distributions:
  $$G_{i,j}^{(g)} \sim \text{Poisson}(\lambda_{i,j})$$
  $$G_{j,i}^{(g)} \sim \text{Poisson}(\lambda_{j,i})$$

  where $G_{i,j}^{(g)}$ represents the number of goals scored by team $i$ against team $j$ in game $g$.

  The log-likelihood contribution for each game is:
  $$\log p(G_{i,j}^{(g)}, G_{j,i}^{(g)} | \phi) = G_{i,j}^{(g)} \cdot \log(\lambda_{i,j}) - \lambda_{i,j} + G_{j,i}^{(g)} \cdot \log(\lambda_{j,i}) - \lambda_{j,i} - \log(G_{i,j}^{(g)}!) - \log(G_{j,i}^{(g)}!)$$

### `poisson_model_2.stan`
- **Description**: Poisson model with home advantage for goal scoring
- **Observed Data**:
  - $N$: total number of games
  - $K$: total number of teams in the competition
  - Team matchups: pairs of teams $(i, j)$ that competed against each other
  - Goal counts: number of goals scored by each team in each game

- **Parameters:**
  - **Team Log-Skills**: Each team has an underlying log-skill level $\phi_i$ that represents their goal-scoring ability relative to other teams.
  - **Home Force**: A parameter $h > 0$ that represents the multiplicative advantage gained by playing at home.
  - **Identifiability Constraint**: To ensure the model is identifiable, we impose a sum-to-zero constraint on the log-skills, meaning they are measured relative to the average team strength.

  **Mathematical Formulation:**
  $$\phi_i \sim \mathcal{N}(0, 1) \quad \text{for } i = 1, \ldots, K-1$$
  $$\phi_K = -\sum_{j=1}^{K-1} \phi_j$$
  $$h \sim \mathcal{N}(1, 1) \quad \text{with } h > 0$$

  This ensures that $\sum_{i=1}^{K} \phi_i = 0$, preventing the model from having infinitely many equivalent solutions.

- **Likelihood:**

  For each game $g$ between team $i$ (home) and team $j$ (away):
  $$\lambda_{i,j} = \exp(\log(\exp(\phi_i) + h) - \phi_j)$$
  $$\lambda_{j,i} = \exp(-\log(\exp(\phi_i) + h) + \phi_j) = \frac{1}{\lambda_{i,j}}$$

  The goal counts follow Poisson distributions:
  $$G_{i,j}^{(g)} \sim \text{Poisson}(\lambda_{i,j})$$
  $$G_{j,i}^{(g)} \sim \text{Poisson}(\lambda_{j,i})$$

  where $G_{i,j}^{(g)}$ represents the number of goals scored by the home team $i$ against the away team $j$ in game $g$.

  The log-likelihood contribution for each game is:
  $$\log p(G_{i,j}^{(g)}, G_{j,i}^{(g)} | \phi, h) = G_{i,j}^{(g)} \cdot \log(\lambda_{i,j}) - \lambda_{i,j} + G_{j,i}^{(g)} \cdot \log(\lambda_{j,i}) - \lambda_{j,i} - \log(G_{i,j}^{(g)}!) - \log(G_{j,i}^{(g)}!)$$
