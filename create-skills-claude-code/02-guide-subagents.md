# Guide complet : Créer des Subagents pour Claude Code

## Table des matières

1. [Introduction](#introduction)
2. [Subagents intégrés](#subagents-intégrés)
3. [Créer son premier Subagent](#créer-son-premier-subagent)
4. [Configurer un Subagent](#configurer-un-subagent)
5. [Choisir le scope](#choisir-le-scope)
6. [Écrire les fichiers Subagent](#écrire-les-fichiers-subagent)
7. [Référence complète du Frontmatter](#référence-complète-du-frontmatter)
8. [Choisir un modèle](#choisir-un-modèle)
9. [Contrôler les capacités](#contrôler-les-capacités)
10. [Modes de permissions](#modes-de-permissions)
11. [Précharger des Skills dans les Subagents](#précharger-des-skills-dans-les-subagents)
12. [Hooks de cycle de vie](#hooks-de-cycle-de-vie)
13. [Mémoire persistante](#mémoire-persistante)
14. [Exécution foreground/background](#exécution-foregroundbackground)
15. [Exemples de Subagents](#exemples-de-subagents)
16. [Lien Skills ↔ Subagents](#lien-skills--subagents)

---

## Introduction

Les **Subagents** sont des assistants IA spécialisés qui gèrent des types de tâches spécifiques. Chaque subagent s'exécute dans sa propre fenêtre de contexte avec :

- Un prompt système personnalisé
- Un accès à des outils spécifiques
- Des permissions indépendantes

Quand Claude rencontre une tâche qui correspond à la description d'un subagent, il lui délègue la tâche. Le subagent travaille indépendamment et retourne les résultats.

### Pourquoi utiliser des Subagents ?

- **Préserver le contexte** : exploration et implémentation hors de la conversation principale
- **Appliquer des contraintes** : limiter les outils utilisables
- **Réutiliser des configurations** : subagents utilisateur disponibles dans tous les projets
- **Spécialiser le comportement** : prompts système focalisés sur des domaines spécifiques
- **Contrôler les coûts** : router les tâches vers des modèles plus rapides/moins chers (Haiku)

---

## Subagents intégrés

Claude Code inclut plusieurs subagents intégrés :

### Explore

Agent rapide, en lecture seule, optimisé pour la recherche et l'analyse de codebases.

- **Modèle** : Haiku (rapide, basse latence)
- **Outils** : Lecture seule (pas d'accès à Write et Edit)
- **Usage** : Découverte de fichiers, recherche de code, exploration de codebase
- **Niveaux** : `quick` (recherche ciblée), `medium` (exploration équilibrée), `very thorough` (analyse complète)

### Plan

Agent de recherche utilisé en mode plan pour rassembler du contexte avant de présenter un plan.

- **Modèle** : Hérite de la conversation principale
- **Outils** : Lecture seule
- **Usage** : Recherche de codebase pour planification

### General-purpose

Agent capable pour les tâches complexes, multi-étapes, nécessitant exploration ET action.

- **Modèle** : Hérite de la conversation principale
- **Outils** : Tous les outils
- **Usage** : Recherche complexe, opérations multi-étapes, modifications de code

### Autres agents intégrés

| Agent | Modèle | Quand Claude l'utilise |
|:---|:---|:---|
| Bash | Hérite | Exécution de commandes terminal dans un contexte séparé |
| statusline-setup | Sonnet | Quand on lance `/statusline` |
| Claude Code Guide | Haiku | Quand on pose des questions sur les fonctionnalités de Claude Code |

---

## Créer son premier Subagent

### Méthode 1 : Via la commande `/agents` (recommandée)

```
/agents
```

1. Sélectionnez **Create new agent**
2. Choisissez **User-level** (disponible dans tous vos projets) ou **Project-level**
3. Sélectionnez **Generate with Claude** et décrivez le subagent
4. Choisissez les outils autorisés
5. Choisissez le modèle
6. Choisissez une couleur
7. Sauvegardez

Le subagent est disponible immédiatement, sans redémarrage.

### Méthode 2 : Manuellement

Créez un fichier markdown dans le bon répertoire :

**Subagent personnel** (tous vos projets) :
```bash
mkdir -p ~/.claude/agents
```

Créez `~/.claude/agents/code-reviewer.md` :

```yaml
---
name: code-reviewer
description: Revoit le code pour la qualité et les bonnes pratiques
tools: Read, Glob, Grep
model: sonnet
---

Tu es un reviewer de code. Quand invoqué, analyse le code et fournis
des retours spécifiques et actionnables sur la qualité, la sécurité
et les bonnes pratiques.
```

**Subagent de projet** (ce projet uniquement) :

```bash
mkdir -p .claude/agents
```

Créez `.claude/agents/code-reviewer.md` avec le même contenu.

> **Note** : Les subagents créés manuellement nécessitent un redémarrage de session ou l'usage de `/agents` pour être chargés immédiatement.

### Méthode 3 : Via le CLI (session uniquement)

```bash
claude --agents '{
  "code-reviewer": {
    "description": "Expert code reviewer. Use proactively after code changes.",
    "prompt": "You are a senior code reviewer. Focus on code quality, security, and best practices.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "sonnet"
  }
}'
```

Le subagent n'existe que pour cette session et n'est pas sauvegardé sur disque.

---

## Configurer un Subagent

### Structure du fichier

```yaml
---
# Frontmatter YAML (configuration)
name: mon-subagent
description: Ce que fait ce subagent
tools: Read, Glob, Grep
model: sonnet
---

# Contenu Markdown (prompt système)

Tu es un assistant spécialisé dans...
```

Le frontmatter définit les métadonnées et la configuration. Le corps devient le prompt système qui guide le comportement du subagent.

> **Important** : Les subagents reçoivent uniquement ce prompt système (plus les détails d'environnement basiques comme le répertoire de travail), PAS le prompt système complet de Claude Code.

---

## Choisir le scope

| Emplacement | Scope | Priorité | Comment créer |
|:---|:---|:---|:---|
| Flag CLI `--agents` | Session courante | 1 (la plus haute) | JSON à la ligne de commande |
| `.claude/agents/` | Projet courant | 2 | Interactif ou manuel |
| `~/.claude/agents/` | Tous vos projets | 3 | Interactif ou manuel |
| Répertoire `agents/` du plugin | Là où le plugin est activé | 4 (la plus basse) | Installé avec les plugins |

Quand plusieurs subagents partagent le même nom, l'emplacement de plus haute priorité gagne.

---

## Écrire les fichiers Subagent

### Exemple complet

```yaml
---
name: safe-researcher
description: Agent de recherche avec des capacités restreintes
tools: Read, Grep, Glob, Bash
disallowedTools: Write, Edit
model: haiku
permissionMode: default
maxTurns: 20
---

Tu es un chercheur de code spécialisé. Tu analyses les codebases
pour comprendre l'architecture et les patterns utilisés.

## Tes responsabilités

1. Explorer la structure du projet
2. Identifier les patterns architecturaux
3. Documenter les dépendances
4. Résumer tes trouvailles de manière claire

## Règles

- Ne modifie JAMAIS de fichiers
- Fournis toujours des références de fichiers spécifiques
- Organise tes trouvailles par catégorie
```

---

## Référence complète du Frontmatter

| Champ | Requis | Description |
|:---|:---|:---|
| `name` | Oui | Identifiant unique (lettres minuscules et tirets) |
| `description` | Oui | Quand Claude devrait déléguer à ce subagent |
| `tools` | Non | Outils que le subagent peut utiliser. Hérite de tous les outils si omis |
| `disallowedTools` | Non | Outils à refuser, retirés de la liste héritée ou spécifiée |
| `model` | Non | Modèle : `sonnet`, `opus`, `haiku`, ou `inherit`. Défaut : `inherit` |
| `permissionMode` | Non | Mode de permissions : `default`, `acceptEdits`, `dontAsk`, `bypassPermissions`, `plan` |
| `maxTurns` | Non | Nombre maximum de tours agentiques avant que le subagent s'arrête |
| `skills` | Non | Skills à charger dans le contexte du subagent au démarrage |
| `mcpServers` | Non | Serveurs MCP disponibles pour ce subagent |
| `hooks` | Non | Hooks de cycle de vie scopés à ce subagent |
| `memory` | Non | Scope de mémoire persistante : `user`, `project`, ou `local` |
| `background` | Non | Mettre à `true` pour toujours exécuter en tâche de fond. Défaut : `false` |
| `isolation` | Non | Mettre à `worktree` pour exécuter dans un git worktree temporaire |

---

## Choisir un modèle

| Valeur | Description |
|:---|:---|
| `sonnet` | Bon équilibre capacité/vitesse |
| `opus` | Le plus capable, plus lent |
| `haiku` | Le plus rapide, le moins cher |
| `inherit` | Même modèle que la conversation principale (défaut) |

---

## Contrôler les capacités

### Outils disponibles

Par défaut, les subagents héritent de tous les outils de la conversation principale, y compris les outils MCP.

**Allowlist** (spécifier les outils autorisés) :
```yaml
tools: Read, Grep, Glob, Bash
```

**Denylist** (retirer des outils spécifiques) :
```yaml
disallowedTools: Write, Edit
```

---

## Modes de permissions

| Mode | Description |
|:---|:---|
| `default` | Hérite du mode de la conversation principale |
| `acceptEdits` | Approuve automatiquement les éditions de fichiers |
| `dontAsk` | N'exécute que les outils auto-approuvés, saute les autres |
| `bypassPermissions` | Approuve tout automatiquement |
| `plan` | Mode lecture seule, pas d'écriture |

---

## Précharger des Skills dans les Subagents

Utilisez le champ `skills` pour injecter le contenu complet de skills dans le prompt système du subagent au démarrage :

```yaml
---
name: frontend-dev
description: Développeur frontend spécialisé
skills:
  - react-patterns
  - testing-conventions
  - api-conventions
model: sonnet
---

Tu es un développeur frontend expert. Utilise les conventions de ce projet
pour écrire du code propre et bien testé.
```

> **Important** : Les subagents n'héritent PAS des skills de la conversation parente. Vous devez les déclarer explicitement via `skills`.

### Différence Skills ↔ Subagents

| Approche | Prompt système | Tâche | Charge aussi |
|:---|:---|:---|:---|
| Skill avec `context: fork` | Du type d'agent (`Explore`, `Plan`, etc.) | Contenu du SKILL.md | CLAUDE.md |
| Subagent avec champ `skills` | Le corps markdown du subagent | Message de délégation de Claude | Skills préchargées + CLAUDE.md |

---

## Hooks de cycle de vie

Les subagents supportent des hooks à des étapes spécifiques :

```yaml
---
name: careful-editor
description: Éditeur prudent avec validation
hooks:
  PreToolUse:
    - matcher: Edit
      hooks:
        - type: command
          command: echo "Edit about to happen"
  PostToolUse:
    - matcher: Edit
      hooks:
        - type: command
          command: npm run lint --fix
---
```

---

## Mémoire persistante

Activez l'apprentissage inter-sessions :

```yaml
---
name: code-reviewer
description: Reviewer de code qui apprend de vos préférences
memory: project
---
```

| Scope | Stockage | Partagé |
|:---|:---|:---|
| `user` | `~/.claude/agent-memory/<name>.md` | Tous vos projets |
| `project` | `.claude/agent-memory/<name>.md` | Membres de l'équipe (si commité) |
| `local` | `.claude/local/agent-memory/<name>.md` | Ce clone uniquement |

---

## Exécution foreground/background

### Foreground (défaut)

Le subagent s'exécute dans le contexte principal. Les résultats sont disponibles immédiatement.

### Background

```yaml
---
name: long-analysis
description: Analyse de longue durée
background: true
---
```

Le subagent s'exécute en arrière-plan. Claude continue la conversation pendant que le subagent travaille.

### Isolation avec worktree

```yaml
---
name: safe-refactor
description: Refactoring isolé
isolation: worktree
---
```

Le subagent obtient une copie isolée du dépôt via git worktree. Le worktree est automatiquement nettoyé si le subagent ne fait aucun changement.

---

## Exemples de Subagents

### Reviewer de sécurité

```yaml
---
name: security-reviewer
description: Analyse le code pour les vulnérabilités de sécurité. Utiliser après des changements de code qui touchent à l'authentification, l'autorisation, ou la gestion de données.
tools: Read, Grep, Glob
model: sonnet
---

Tu es un expert en sécurité applicative. Analyse le code pour :

1. **Injection** : SQL injection, command injection, XSS
2. **Authentification** : Failles dans les flux d'auth
3. **Autorisation** : Élévation de privilèges, IDOR
4. **Données sensibles** : Secrets en dur, logging de données sensibles
5. **Dépendances** : Bibliothèques avec des CVE connues

Pour chaque trouvaille :
- Décris la vulnérabilité
- Évalue la sévérité (Critique/Haute/Moyenne/Basse)
- Propose une correction concrète
```

### Générateur de tests

```yaml
---
name: test-generator
description: Génère des tests unitaires et d'intégration pour le code existant
tools: Read, Grep, Glob, Write, Edit, Bash
model: sonnet
skills:
  - testing-conventions
---

Tu es un expert en testing. Quand on te demande de générer des tests :

1. Lis le code source à tester
2. Identifie les cas de test (happy path, edge cases, error cases)
3. Génère des tests complets en suivant les conventions du projet
4. Exécute les tests pour vérifier qu'ils passent
5. Corrige tout test qui échoue
```

### Documenteur d'API

```yaml
---
name: api-documenter
description: Génère de la documentation API à partir du code source
tools: Read, Grep, Glob, Write
model: haiku
---

Tu es un rédacteur technique spécialisé en documentation API.

1. Analyse les endpoints/routes du code source
2. Extrais les paramètres, types de retour, et codes d'erreur
3. Génère une documentation claire avec des exemples
4. Utilise le format OpenAPI/Swagger quand approprié
```

### Agent de migration

```yaml
---
name: migrator
description: Aide à migrer du code entre versions de frameworks ou de langages
tools: Read, Grep, Glob, Write, Edit, Bash
model: opus
isolation: worktree
---

Tu es un expert en migration de code. Tu travailles dans un worktree isolé
pour éviter de casser le code principal.

1. Analyse le code existant
2. Identifie les patterns à migrer
3. Applique les changements de manière systématique
4. Vérifie que les tests passent toujours
5. Résume tous les changements effectués
```
