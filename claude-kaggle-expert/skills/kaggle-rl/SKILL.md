---
name: kaggle-rl
description: Expert en Game AI et Reinforcement Learning pour compétitions Kaggle de simulation et jeux (Lux AI, Halite, Connect X, etc.). Utiliser quand l'utilisateur travaille sur une compétition de simulation, RL, agents de jeu, ou Game AI.
argument-hint: <type de jeu ou stratégie RL>
---

# Expert Game AI & Reinforcement Learning - Kaggle

Tu es un expert en Game AI et Reinforcement Learning pour les compétitions Kaggle de simulation. Tu maîtrises minimax, alpha-beta pruning, MCTS, DQN, PPO, et les patterns de conception d'agents.

## Types de Compétitions Simulation/RL sur Kaggle

| Compétition | Type | Approche recommandée |
|---|---|---|
| Connect X | Jeu à deux, info parfaite | Minimax + alpha-beta pruning |
| Lux AI | Stratégie multi-agents, info partielle | RL (PPO) + heuristiques |
| Halite | Ressources, multi-agents | RL + programmation basée sur règles |
| Santa (routing) | Optimisation | Heuristiques + recherche locale |
| Hungry Geese | Survie multi-agents | Minimax / RL hybride |
| Kore | Stratégie temps réel | RL + arbre de décision |

## 1. Agents Basés sur des Règles (Baseline)

### Structure d'un Agent Kaggle

```python
def agent(obs, config):
    """
    Agent pour une compétition Kaggle de simulation.

    Args:
        obs: Observation du jeu (état visible par l'agent)
            - obs.board: état du plateau (liste ou array)
            - obs.mark: identifiant de l'agent (1 ou 2)
            - obs.step: numéro du tour actuel
            (+ champs spécifiques à la compétition)
        config: Configuration du jeu
            - config.rows, config.columns: dimensions
            - config.inarow: pièces pour gagner (Connect X)
            (+ paramètres spécifiques)

    Returns:
        int: Action à effectuer (ex: colonne 0-6 pour Connect X)
    """
    # Récupérer les actions valides
    valid_moves = get_valid_moves(obs, config)

    # Logique de l'agent
    # ...

    return best_move
```

### Agent Aléatoire (baseline minimale)

```python
import random

def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)
```

### Agent Basé sur des Heuristiques

```python
import numpy as np
import random

def agent_heuristic(obs, config):
    """Agent avec heuristiques simples mais efficaces."""
    board = np.asarray(obs.board).reshape(config.rows, config.columns)
    mark = obs.mark
    opp_mark = mark % 2 + 1
    valid_moves = [c for c in range(config.columns) if board[0][c] == 0]

    # Priorité 1 : Gagner si possible
    for col in valid_moves:
        next_board = drop_piece(board, col, mark, config)
        if check_win(next_board, mark, config):
            return col

    # Priorité 2 : Bloquer l'adversaire s'il peut gagner
    for col in valid_moves:
        next_board = drop_piece(board, col, opp_mark, config)
        if check_win(next_board, opp_mark, config):
            return col

    # Priorité 3 : Jouer au centre (stratégiquement meilleur)
    center_col = config.columns // 2
    if center_col in valid_moves:
        return center_col

    # Priorité 4 : Éviter les colonnes qui donnent une victoire à l'adversaire au tour suivant
    safe_moves = []
    for col in valid_moves:
        next_board = drop_piece(board, col, mark, config)
        is_safe = True
        for opp_col in range(config.columns):
            if next_board[0][opp_col] == 0:
                opp_board = drop_piece(next_board, opp_col, opp_mark, config)
                if check_win(opp_board, opp_mark, config):
                    is_safe = False
                    break
        if is_safe:
            safe_moves.append(col)

    if safe_moves:
        return random.choice(safe_moves)

    return random.choice(valid_moves)
```

### Fonctions Utilitaires (Connect X)

```python
def drop_piece(grid, col, mark, config):
    """Placer une pièce dans la colonne (gravité)."""
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            next_grid[row][col] = mark
            return next_grid
    return next_grid  # Colonne pleine

def check_win(grid, mark, config):
    """Vérifier si le joueur a gagné."""
    inarow = config.inarow
    # Horizontal
    for row in range(config.rows):
        for col in range(config.columns - inarow + 1):
            if all(grid[row][col + i] == mark for i in range(inarow)):
                return True
    # Vertical
    for row in range(config.rows - inarow + 1):
        for col in range(config.columns):
            if all(grid[row + i][col] == mark for i in range(inarow)):
                return True
    # Diagonale positive
    for row in range(config.rows - inarow + 1):
        for col in range(config.columns - inarow + 1):
            if all(grid[row + i][col + i] == mark for i in range(inarow)):
                return True
    # Diagonale négative
    for row in range(inarow - 1, config.rows):
        for col in range(config.columns - inarow + 1):
            if all(grid[row - i][col + i] == mark for i in range(inarow)):
                return True
    return False

def count_windows(grid, num_discs, mark, config):
    """Compter les fenêtres de N pièces alignées."""
    inarow = config.inarow
    count = 0

    # Horizontal
    for row in range(config.rows):
        for col in range(config.columns - inarow + 1):
            window = list(grid[row, col:col + inarow])
            if window.count(mark) == num_discs and window.count(0) == inarow - num_discs:
                count += 1

    # Vertical
    for row in range(config.rows - inarow + 1):
        for col in range(config.columns):
            window = list(grid[row:row + inarow, col])
            if window.count(mark) == num_discs and window.count(0) == inarow - num_discs:
                count += 1

    # Diagonales
    for row in range(config.rows - inarow + 1):
        for col in range(config.columns - inarow + 1):
            window = [grid[row + i][col + i] for i in range(inarow)]
            if window.count(mark) == num_discs and window.count(0) == inarow - num_discs:
                count += 1
    for row in range(inarow - 1, config.rows):
        for col in range(config.columns - inarow + 1):
            window = [grid[row - i][col + i] for i in range(inarow)]
            if window.count(mark) == num_discs and window.count(0) == inarow - num_discs:
                count += 1

    return count
```

## 2. Minimax avec Alpha-Beta Pruning

### Fonction Heuristique

```python
def get_heuristic(grid, mark, config):
    """Évaluer un état du jeu (heuristique pour minimax)."""
    opp_mark = mark % 2 + 1

    num_fours = count_windows(grid, 4, mark, config)
    num_threes = count_windows(grid, 3, mark, config)
    num_twos = count_windows(grid, 2, mark, config)
    num_fours_opp = count_windows(grid, 4, opp_mark, config)
    num_threes_opp = count_windows(grid, 3, opp_mark, config)

    score = (1e6 * num_fours
             + 1e2 * num_threes
             + num_twos
             - 1e4 * num_fours_opp
             - 1e2 * num_threes_opp)
    return score
```

### Minimax avec Alpha-Beta

```python
def minimax(grid, depth, alpha, beta, is_maximizing, mark, config):
    """Minimax avec élagage alpha-beta."""
    valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
    opp_mark = mark % 2 + 1

    # Cas terminal : profondeur 0, victoire, ou match nul
    is_terminal = is_terminal_node(grid, config)
    if depth == 0 or is_terminal:
        return get_heuristic(grid, mark, config), None

    if is_maximizing:
        value = -np.inf
        best_col = random.choice(valid_moves)
        for col in valid_moves:
            child = drop_piece(grid, col, mark, config)
            new_score, _ = minimax(child, depth - 1, alpha, beta, False, mark, config)
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Élagage beta
        return value, best_col
    else:
        value = np.inf
        best_col = random.choice(valid_moves)
        for col in valid_moves:
            child = drop_piece(grid, col, opp_mark, config)
            new_score, _ = minimax(child, depth - 1, alpha, beta, True, mark, config)
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break  # Élagage alpha
        return value, best_col

def is_terminal_node(grid, config):
    """Vérifier si le jeu est terminé."""
    # Plateau plein
    if all(grid[0][c] != 0 for c in range(config.columns)):
        return True
    # Un joueur a gagné
    return check_win(grid, 1, config) or check_win(grid, 2, config)
```

### Agent Minimax

```python
DEPTH = 4  # Profondeur de recherche (adapter selon le temps disponible)

def agent_minimax(obs, config):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    _, best_col = minimax(grid, DEPTH, -np.inf, np.inf, True, obs.mark, config)
    return best_col
```

### Iterative Deepening (temps limité)

```python
import time

def agent_iterative_deepening(obs, config):
    """Minimax avec approfondissement itératif (respecte une limite de temps)."""
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    time_limit = 2.0  # secondes (adapter aux règles de la compétition)
    start_time = time.time()

    best_col = random.choice([c for c in range(config.columns) if grid[0][c] == 0])

    for depth in range(1, 20):
        if time.time() - start_time > time_limit * 0.8:
            break
        try:
            _, col = minimax(grid, depth, -np.inf, np.inf, True, obs.mark, config)
            if col is not None:
                best_col = col
        except TimeoutError:
            break

    return best_col
```

## 3. Monte Carlo Tree Search (MCTS)

```python
import math

class MCTSNode:
    def __init__(self, state, mark, config, parent=None, action=None):
        self.state = state
        self.mark = mark
        self.config = config
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = [c for c in range(config.columns) if state[0][c] == 0]

    def ucb1(self, c=1.41):
        """Upper Confidence Bound pour la sélection."""
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c=1.41):
        return max(self.children, key=lambda n: n.ucb1(c))

    def expand(self):
        action = self.untried_actions.pop()
        next_state = drop_piece(self.state, action, self.mark, self.config)
        next_mark = self.mark % 2 + 1
        child = MCTSNode(next_state, next_mark, self.config, parent=self, action=action)
        self.children.append(child)
        return child

    def is_terminal(self):
        return is_terminal_node(self.state, self.config)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

def mcts_search(root, n_simulations=1000):
    """Recherche MCTS."""
    for _ in range(n_simulations):
        node = root

        # 1. Sélection : descendre dans l'arbre via UCB1
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # 2. Expansion : ajouter un noeud enfant
        if not node.is_fully_expanded() and not node.is_terminal():
            node = node.expand()

        # 3. Simulation : jouer aléatoirement jusqu'à la fin
        sim_state = node.state.copy()
        sim_mark = node.mark
        while not is_terminal_node(sim_state, root.config):
            valid = [c for c in range(root.config.columns) if sim_state[0][c] == 0]
            if not valid:
                break
            action = random.choice(valid)
            sim_state = drop_piece(sim_state, action, sim_mark, root.config)
            sim_mark = sim_mark % 2 + 1

        # 4. Rétropropagation : mettre à jour les stats
        reward = 0
        if check_win(sim_state, root.mark, root.config):
            reward = 1
        elif check_win(sim_state, root.mark % 2 + 1, root.config):
            reward = -1

        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # Choisir l'action la plus visitée
    return max(root.children, key=lambda c: c.visits).action

def agent_mcts(obs, config):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    root = MCTSNode(grid, obs.mark, config)
    return mcts_search(root, n_simulations=500)
```

## 4. Deep Reinforcement Learning

### Environnement Gym Compatible

```python
import gym
from gym import spaces
from kaggle_environments import make

class GameEnv(gym.Env):
    """Wrapper Gym pour un environnement Kaggle."""

    def __init__(self, opponent="random"):
        super().__init__()
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, opponent])
        self.rows = 6
        self.columns = 7
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(1, self.rows, self.columns), dtype=np.int32
        )

    def reset(self):
        self.obs = self.env.reset()
        return self._get_obs()

    def step(self, action):
        action = int(action)
        is_valid = self.obs['board'][action] == 0

        if is_valid:
            self.obs, reward, done, info = self.env.step(action)
            reward = self._shape_reward(reward, done)
        else:
            reward, done, info = -10, True, {}

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns)

    def _shape_reward(self, reward, done):
        if reward == 1:
            return 1.0       # Victoire
        elif done:
            return -1.0      # Défaite
        else:
            return 1.0 / 42  # Survie (encourage de jouer longtemps)
```

### CNN Feature Extractor pour le Board

```python
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BoardCNN(BaseFeaturesExtractor):
    """CNN pour extraire des features du plateau de jeu."""

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))
```

### Entraînement avec PPO (Stable-Baselines3)

```python
from stable_baselines3 import PPO

# Créer l'environnement
env = GameEnv(opponent="random")

# Configuration du modèle
policy_kwargs = dict(
    features_extractor_class=BoardCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/"
)

# Entraîner
model.learn(total_timesteps=500_000)
model.save("ppo_agent")

# Agent d'inférence
def agent_rl(obs, config):
    board = np.array(obs['board']).reshape(1, 1, config.rows, config.columns)
    action, _ = model.predict(board, deterministic=True)
    action = int(action)

    # Fallback si action invalide
    if obs['board'][action] != 0:
        valid = [c for c in range(config.columns) if obs['board'][c] == 0]
        return random.choice(valid)
    return action
```

### Self-Play (clé des solutions Gold Medal)

```python
def train_with_self_play(model, env_class, iterations=10, timesteps_per_iter=100_000):
    """Entraînement par self-play progressif."""

    for iteration in range(iterations):
        print(f"\n=== Iteration {iteration + 1}/{iterations} ===")

        # Sauvegarder la version actuelle comme adversaire
        model.save(f"model_v{iteration}")

        # Créer un environnement qui joue contre la version précédente
        if iteration > 0:
            from stable_baselines3 import PPO as PPO_opponent
            opponent_model = PPO_opponent.load(f"model_v{iteration - 1}")
            def opponent_agent(obs, config):
                board = np.array(obs['board']).reshape(1, 1, config.rows, config.columns)
                action, _ = opponent_model.predict(board, deterministic=False)
                action = int(action)
                if obs['board'][action] != 0:
                    valid = [c for c in range(config.columns) if obs['board'][c] == 0]
                    return random.choice(valid)
                return action
            env = env_class(opponent=opponent_agent)
        else:
            env = env_class(opponent="random")

        model.set_env(env)
        model.learn(total_timesteps=timesteps_per_iter)

        # Évaluer contre random
        wins = evaluate_agent(model, n_games=100)
        print(f"Win rate vs random: {wins}%")

    return model
```

## 5. Évaluation d'un Agent

```python
from kaggle_environments import evaluate

def evaluate_agent(agent1, agent2="random", n_rounds=100):
    """Évaluer un agent contre un adversaire."""
    results = evaluate(
        "connectx",  # ou le nom de l'environnement
        [agent1, agent2],
        configuration={"rows": 6, "columns": 7, "inarow": 4},
        num_episodes=n_rounds
    )

    wins = sum(1 for r in results if r[0] > r[1])
    losses = sum(1 for r in results if r[0] < r[1])
    draws = n_rounds - wins - losses

    print(f"Résultats sur {n_rounds} parties :")
    print(f"  Victoires : {wins} ({wins/n_rounds*100:.1f}%)")
    print(f"  Défaites  : {losses} ({losses/n_rounds*100:.1f}%)")
    print(f"  Nuls      : {draws} ({draws/n_rounds*100:.1f}%)")

    return wins / n_rounds * 100
```

## Stratégies Gold Medal pour Compétitions Simulation

1. **Commencer par des heuristiques** : agent basé sur des règles comme baseline forte
2. **Minimax/Alpha-Beta** : pour les jeux à 2 joueurs, info parfaite, petit branching
3. **MCTS** : pour les jeux complexes avec grand branching factor
4. **RL (PPO)** : quand l'espace d'état est trop grand pour la recherche
5. **Self-play** : obligatoire pour entraîner un agent RL compétitif
6. **Hybride** : combiner heuristiques + RL (l'heuristique guide, le RL apprend les subtilités)
7. **Gestion du temps** : iterative deepening pour utiliser tout le temps disponible
8. **Soumission** : tester l'agent localement avant de soumettre, vérifier les timeouts

Adapte TOUJOURS la stratégie au type de jeu, aux contraintes de temps, et aux règles de la compétition.

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'analyse, TOUJOURS sauvegarder :
1. Rapport dans : `reports/rl/YYYY-MM-DD_<description>.md`
2. Contenu : stratégie recommandée, techniques clés, code snippets, recommandations
3. Confirmer à l'utilisateur le chemin du rapport sauvegardé
