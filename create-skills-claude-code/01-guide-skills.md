# Guide complet : Créer des Skills pour Claude Code

## Table des matières

1. [Introduction](#introduction)
2. [Qu'est-ce qu'une Skill ?](#quest-ce-quune-skill)
3. [Créer sa première Skill](#créer-sa-première-skill)
4. [Où stocker les Skills](#où-stocker-les-skills)
5. [Configurer une Skill (Frontmatter YAML)](#configurer-une-skill-frontmatter-yaml)
6. [Référence complète du Frontmatter](#référence-complète-du-frontmatter)
7. [Types de contenu](#types-de-contenu)
8. [Contrôler qui invoque la Skill](#contrôler-qui-invoque-la-skill)
9. [Passer des arguments](#passer-des-arguments)
10. [Substitutions de variables](#substitutions-de-variables)
11. [Ajouter des fichiers de support](#ajouter-des-fichiers-de-support)
12. [Patterns avancés](#patterns-avancés)
13. [Restreindre l'accès aux outils](#restreindre-laccès-aux-outils)
14. [Restreindre l'accès de Claude aux Skills](#restreindre-laccès-de-claude-aux-skills)
15. [Partager des Skills](#partager-des-skills)
16. [Dépannage](#dépannage)

---

## Introduction

Les **Skills** étendent les capacités de Claude Code. Elles permettent de créer des commandes personnalisées, d'ajouter des connaissances spécifiques, et d'automatiser des workflows récurrents.

Une Skill est simplement un fichier `SKILL.md` contenant des instructions. Claude l'ajoute à sa boîte à outils et l'utilise quand c'est pertinent, ou vous pouvez l'invoquer directement avec `/nom-de-la-skill`.

> **Note** : Les Skills suivent le standard ouvert [Agent Skills](https://agentskills.io), compatible avec plusieurs outils IA. Claude Code y ajoute des fonctionnalités comme le contrôle d'invocation, l'exécution en sous-agent, et l'injection de contexte dynamique.

> **Note** : Les anciens "custom slash commands" (`.claude/commands/`) ont été fusionnés dans les Skills. Vos fichiers `.claude/commands/` existants continuent de fonctionner. Si une skill et une commande partagent le même nom, la skill a priorité.

---

## Qu'est-ce qu'une Skill ?

Une Skill est composée de :

1. **Un fichier `SKILL.md`** (obligatoire) : contient les instructions principales
2. **Un frontmatter YAML** (optionnel) : configure le comportement de la skill
3. **Des fichiers de support** (optionnels) : templates, exemples, scripts

**Structure d'une Skill :**

```
ma-skill/
├── SKILL.md           # Instructions principales (obligatoire)
├── template.md        # Template à remplir par Claude
├── examples/
│   └── sample.md      # Exemple de sortie attendue
└── scripts/
    └── validate.sh    # Script exécutable par Claude
```

---

## Créer sa première Skill

### Étape 1 : Créer le répertoire

```bash
# Skill personnelle (disponible dans tous vos projets)
mkdir -p ~/.claude/skills/explain-code

# OU skill de projet (disponible uniquement dans ce projet)
mkdir -p .claude/skills/explain-code
```

### Étape 2 : Écrire le fichier SKILL.md

Créez `SKILL.md` dans le répertoire :

```yaml
---
name: explain-code
description: Explique du code avec des diagrammes visuels et des analogies. À utiliser quand on explique comment du code fonctionne ou quand l'utilisateur demande "comment ça marche ?"
---

Quand tu expliques du code, inclus toujours :

1. **Commence par une analogie** : Compare le code à quelque chose de la vie quotidienne
2. **Dessine un diagramme** : Utilise de l'ASCII art pour montrer le flux, la structure ou les relations
3. **Parcours le code** : Explique étape par étape ce qui se passe
4. **Signale un piège** : Quel est l'erreur courante ou l'idée reçue ?

Garde les explications conversationnelles. Pour les concepts complexes, utilise plusieurs analogies.
```

### Étape 3 : Tester la Skill

**Invocation automatique** (Claude la charge quand c'est pertinent) :
```
Comment est-ce que ce code fonctionne ?
```

**Invocation directe** avec le nom de la skill :
```
/explain-code src/auth/login.ts
```

---

## Où stocker les Skills

L'emplacement détermine qui peut utiliser la skill :

| Emplacement | Chemin | S'applique à |
|:---|:---|:---|
| **Enterprise** | Via les managed settings | Tous les utilisateurs de l'organisation |
| **Personnel** | `~/.claude/skills/<nom>/SKILL.md` | Tous vos projets |
| **Projet** | `.claude/skills/<nom>/SKILL.md` | Ce projet uniquement |
| **Plugin** | `<plugin>/skills/<nom>/SKILL.md` | Là où le plugin est activé |

**Priorité** (du plus élevé au plus bas) : Enterprise > Personnel > Projet

Les skills de plugins utilisent un namespace `plugin-name:skill-name`, donc elles ne peuvent pas entrer en conflit avec les autres niveaux.

### Découverte automatique dans les sous-répertoires

Claude Code découvre automatiquement les skills dans les répertoires `.claude/skills/` imbriqués. Par exemple, si vous éditez un fichier dans `packages/frontend/`, Claude cherche aussi dans `packages/frontend/.claude/skills/`. Cela supporte les monorepos.

### Skills depuis des répertoires additionnels

Les skills dans `.claude/skills/` des répertoires ajoutés via `--add-dir` sont chargées automatiquement avec détection des changements en temps réel.

---

## Configurer une Skill (Frontmatter YAML)

Le frontmatter YAML se place entre des marqueurs `---` en haut du fichier `SKILL.md` :

```yaml
---
name: ma-skill
description: Ce que fait cette skill
disable-model-invocation: true
allowed-tools: Read, Grep
context: fork
agent: Explore
model: sonnet
---

Instructions de la skill ici...
```

---

## Référence complète du Frontmatter

| Champ | Requis | Description |
|:---|:---|:---|
| `name` | Non | Nom d'affichage. Si omis, utilise le nom du répertoire. Lettres minuscules, chiffres et tirets uniquement (max 64 caractères). |
| `description` | Recommandé | Ce que fait la skill et quand l'utiliser. Claude l'utilise pour décider quand la charger automatiquement. Si omis, utilise le premier paragraphe du contenu markdown. |
| `argument-hint` | Non | Indication affichée lors de l'autocomplétion. Ex : `[issue-number]` ou `[filename] [format]`. |
| `disable-model-invocation` | Non | Mettre à `true` pour empêcher Claude de charger cette skill automatiquement. Utilisez pour les workflows à déclencher manuellement avec `/nom`. Défaut : `false`. |
| `user-invocable` | Non | Mettre à `false` pour masquer du menu `/`. Pour les connaissances de fond que les utilisateurs ne devraient pas invoquer directement. Défaut : `true`. |
| `allowed-tools` | Non | Outils que Claude peut utiliser sans demander permission quand cette skill est active. |
| `model` | Non | Modèle à utiliser quand cette skill est active. |
| `context` | Non | Mettre à `fork` pour exécuter dans un contexte de sous-agent isolé. |
| `agent` | Non | Type de sous-agent à utiliser quand `context: fork` est défini. |
| `hooks` | Non | Hooks scopés au cycle de vie de cette skill. |

---

## Types de contenu

### Contenu de référence

Ajoute des connaissances que Claude applique à votre travail courant. S'exécute en ligne.

```yaml
---
name: api-conventions
description: Patterns de design API pour ce codebase
---

Quand tu écris des endpoints API :
- Utilise les conventions de nommage RESTful
- Retourne des formats d'erreur cohérents
- Inclus la validation des requêtes
```

### Contenu de tâche

Instructions étape par étape pour une action spécifique. Souvent invoqué manuellement.

```yaml
---
name: deploy
description: Déployer l'application en production
context: fork
disable-model-invocation: true
---

Déploie l'application :
1. Lance la suite de tests
2. Build l'application
3. Push vers la cible de déploiement
```

---

## Contrôler qui invoque la Skill

Par défaut, vous ET Claude pouvez invoquer n'importe quelle skill. Deux champs frontmatter permettent de restreindre cela :

### `disable-model-invocation: true` — Seul l'utilisateur peut invoquer

Pour les workflows avec des effets de bord ou dont vous voulez contrôler le timing : `/commit`, `/deploy`, `/send-slack-message`.

```yaml
---
name: deploy
description: Déployer l'application en production
disable-model-invocation: true
---

Déploie $ARGUMENTS en production :
1. Lance la suite de tests
2. Build l'application
3. Push vers la cible de déploiement
4. Vérifie que le déploiement a réussi
```

### `user-invocable: false` — Seul Claude peut invoquer

Pour les connaissances de fond qui ne sont pas des actions utilisateur. Ex : un skill `legacy-system-context` qui explique comment un ancien système fonctionne.

### Tableau récapitulatif

| Frontmatter | Utilisateur peut invoquer | Claude peut invoquer | Chargement en contexte |
|:---|:---|:---|:---|
| (défaut) | Oui | Oui | Description toujours en contexte, skill complète chargée à l'invocation |
| `disable-model-invocation: true` | Oui | Non | Description pas en contexte, skill complète chargée quand vous l'invoquez |
| `user-invocable: false` | Non | Oui | Description toujours en contexte, skill complète chargée à l'invocation |

---

## Passer des arguments

Les arguments sont disponibles via le placeholder `$ARGUMENTS`.

### Argument unique

```yaml
---
name: fix-issue
description: Corriger un issue GitHub
disable-model-invocation: true
---

Corrige l'issue GitHub $ARGUMENTS en suivant nos standards de code.

1. Lis la description de l'issue
2. Comprends les exigences
3. Implémente la correction
4. Écris les tests
5. Crée un commit
```

Usage : `/fix-issue 123` → Claude reçoit "Corrige l'issue GitHub 123..."

### Arguments positionnels

Utilisez `$ARGUMENTS[N]` ou le raccourci `$N` :

```yaml
---
name: migrate-component
description: Migrer un composant d'un framework à un autre
---

Migre le composant $0 de $1 vers $2.
Préserve tout le comportement existant et les tests.
```

Usage : `/migrate-component SearchBar React Vue`

> **Note** : Si vous invoquez une skill avec des arguments mais que la skill ne contient pas `$ARGUMENTS`, Claude Code ajoute `ARGUMENTS: <votre input>` à la fin du contenu.

---

## Substitutions de variables

| Variable | Description |
|:---|:---|
| `$ARGUMENTS` | Tous les arguments passés lors de l'invocation |
| `$ARGUMENTS[N]` | Accès à un argument spécifique par index (base 0) |
| `$N` | Raccourci pour `$ARGUMENTS[N]` |
| `${CLAUDE_SESSION_ID}` | L'ID de session courante |

**Exemple :**

```yaml
---
name: session-logger
description: Logger l'activité de cette session
---

Log ce qui suit dans logs/${CLAUDE_SESSION_ID}.log :

$ARGUMENTS
```

---

## Ajouter des fichiers de support

Les skills peuvent inclure plusieurs fichiers dans leur répertoire :

```
ma-skill/
├── SKILL.md           (obligatoire - vue d'ensemble et navigation)
├── reference.md       (docs API détaillées - chargé au besoin)
├── examples.md        (exemples d'utilisation - chargé au besoin)
└── scripts/
    └── helper.py      (script utilitaire - exécuté, pas chargé)
```

Référencez les fichiers de support depuis `SKILL.md` :

```markdown
## Ressources additionnelles

- Pour les détails complets de l'API, voir [reference.md](reference.md)
- Pour les exemples d'utilisation, voir [examples.md](examples.md)
```

> **Conseil** : Gardez `SKILL.md` sous 500 lignes. Déplacez la documentation détaillée dans des fichiers séparés.

---

## Patterns avancés

### Injecter du contexte dynamique

La syntaxe `` !`commande` `` exécute des commandes shell avant que le contenu soit envoyé à Claude. La sortie de la commande remplace le placeholder.

```yaml
---
name: pr-summary
description: Résumer les changements dans une pull request
context: fork
agent: Explore
allowed-tools: Bash(gh *)
---

## Contexte de la pull request
- Diff PR : !`gh pr diff`
- Commentaires PR : !`gh pr view --comments`
- Fichiers modifiés : !`gh pr diff --name-only`

## Ta tâche
Résume cette pull request...
```

**Fonctionnement :**
1. Chaque `` !`commande` `` s'exécute immédiatement (avant que Claude ne voie quoi que ce soit)
2. La sortie remplace le placeholder dans le contenu de la skill
3. Claude reçoit le prompt entièrement rendu avec les données réelles

### Exécuter des Skills dans un sous-agent

Ajoutez `context: fork` au frontmatter pour exécuter une skill de manière isolée :

```yaml
---
name: deep-research
description: Rechercher un sujet en profondeur
context: fork
agent: Explore
---

Recherche $ARGUMENTS en profondeur :

1. Trouve les fichiers pertinents avec Glob et Grep
2. Lis et analyse le code
3. Résume les trouvailles avec des références de fichiers spécifiques
```

**Quand la skill s'exécute :**
1. Un nouveau contexte isolé est créé
2. Le sous-agent reçoit le contenu de la skill comme prompt
3. Le champ `agent` détermine l'environnement d'exécution (modèle, outils, permissions)
4. Les résultats sont résumés et retournés à la conversation principale

**Options d'agent** : `Explore`, `Plan`, `general-purpose`, ou tout sous-agent personnalisé depuis `.claude/agents/`.

> **Attention** : `context: fork` n'a de sens que pour les skills avec des instructions explicites. Si votre skill contient des guidelines sans tâche, le sous-agent n'a rien d'actionnable.

### Activer le thinking étendu

Incluez le mot "ultrathink" n'importe où dans le contenu de votre skill pour activer l'extended thinking.

### Générer des sorties visuelles

Les skills peuvent embarquer et exécuter des scripts dans n'importe quel langage. Un pattern puissant est la génération de fichiers HTML interactifs :

```yaml
---
name: codebase-visualizer
description: Génère une visualisation interactive en arbre de votre codebase
allowed-tools: Bash(python *)
---

# Codebase Visualizer

Génère une vue HTML interactive en arbre montrant la structure de fichiers du projet.

## Utilisation

Lance le script de visualisation depuis la racine du projet :

\`\`\`bash
python ~/.claude/skills/codebase-visualizer/scripts/visualize.py .
\`\`\`

Cela crée `codebase-map.html` dans le répertoire courant et l'ouvre dans le navigateur.
```

---

## Restreindre l'accès aux outils

Utilisez `allowed-tools` pour limiter les outils disponibles :

```yaml
---
name: safe-reader
description: Lire des fichiers sans faire de modifications
allowed-tools: Read, Grep, Glob
---
```

---

## Restreindre l'accès de Claude aux Skills

### Désactiver toutes les skills

Ajoutez dans les règles de refus de `/permissions` :
```
Skill
```

### Autoriser/refuser des skills spécifiques

```
# Autoriser uniquement des skills spécifiques
Skill(commit)
Skill(review-pr *)

# Refuser des skills spécifiques
Skill(deploy *)
```

Syntaxe : `Skill(name)` pour correspondance exacte, `Skill(name *)` pour correspondance par préfixe.

### Masquer des skills individuelles

Ajoutez `disable-model-invocation: true` au frontmatter pour retirer la skill du contexte de Claude.

---

## Partager des Skills

| Méthode | Comment |
|:---|:---|
| **Skills de projet** | Committez `.claude/skills/` dans le contrôle de version |
| **Plugins** | Créez un répertoire `skills/` dans votre plugin |
| **Managed** | Déployez à l'échelle de l'organisation via les managed settings |

---

## Dépannage

### La skill ne se déclenche pas

1. Vérifiez que la description inclut des mots-clés que les utilisateurs diraient naturellement
2. Vérifiez que la skill apparaît dans "Quelles skills sont disponibles ?"
3. Reformulez votre requête pour correspondre davantage à la description
4. Invoquez-la directement avec `/nom-de-la-skill`

### La skill se déclenche trop souvent

1. Rendez la description plus spécifique
2. Ajoutez `disable-model-invocation: true` pour n'autoriser que l'invocation manuelle

### Claude ne voit pas toutes mes skills

Les descriptions de skills sont chargées en contexte. Si vous avez beaucoup de skills, elles peuvent dépasser le budget de caractères. Le budget s'adapte dynamiquement à 2% de la fenêtre de contexte, avec un fallback de 16 000 caractères.

Lancez `/context` pour vérifier s'il y a un avertissement sur les skills exclues.

Pour surcharger la limite, définissez la variable d'environnement `SLASH_COMMAND_TOOL_CHAR_BUDGET`.
