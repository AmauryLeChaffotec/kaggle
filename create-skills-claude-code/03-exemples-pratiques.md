# Exemples pratiques de Skills et Subagents

Ce fichier contient des exemples complets, prêts à l'emploi, pour différents cas d'usage.

---

## Sommaire

1. [Skills de développement](#1-skills-de-développement)
2. [Skills de workflow Git](#2-skills-de-workflow-git)
3. [Skills de qualité de code](#3-skills-de-qualité-de-code)
4. [Skills de documentation](#4-skills-de-documentation)
5. [Skills avec scripts embarqués](#5-skills-avec-scripts-embarqués)
6. [Skills avec contexte dynamique](#6-skills-avec-contexte-dynamique)
7. [Subagents spécialisés](#7-subagents-spécialisés)

---

## 1. Skills de développement

### Créer un composant React

**Chemin** : `.claude/skills/create-component/SKILL.md`

```yaml
---
name: create-component
description: Crée un composant React avec TypeScript, tests et styles
argument-hint: [ComponentName]
disable-model-invocation: true
---

Crée un nouveau composant React nommé $0 :

## Structure à créer

- `src/components/$0/$0.tsx` - Le composant
- `src/components/$0/$0.test.tsx` - Les tests
- `src/components/$0/$0.module.css` - Les styles CSS modules
- `src/components/$0/index.ts` - Le barrel export

## Conventions

- Utilise des composants fonctionnels avec TypeScript
- Exporte le type des props
- Ajoute au minimum 3 tests : rendu, props, interaction
- Utilise CSS modules pour le styling
- Le barrel export doit exporter le composant et ses types

## Template du composant

\`\`\`tsx
import React from 'react';
import styles from './$0.module.css';

export interface $0Props {
  // Props ici
}

export const $0: React.FC<$0Props> = (props) => {
  return (
    <div className={styles.container}>
      {/* Contenu */}
    </div>
  );
};
\`\`\`
```

**Usage** : `/create-component UserProfile`

---

### Créer un endpoint API

**Chemin** : `.claude/skills/create-endpoint/SKILL.md`

```yaml
---
name: create-endpoint
description: Crée un endpoint API REST avec validation, tests et documentation
argument-hint: [method] [path]
disable-model-invocation: true
---

Crée un nouvel endpoint API : $0 $1

## Étapes

1. **Identifie le framework** utilisé dans le projet (Express, Fastify, Next.js, etc.)
2. **Crée le handler** avec :
   - Validation des entrées (paramètres, body, query)
   - Gestion d'erreurs appropriée
   - Codes HTTP corrects
   - Types TypeScript si applicable
3. **Crée les tests** :
   - Test du happy path
   - Test de validation (entrées invalides)
   - Test des erreurs (404, 500)
4. **Mets à jour le routeur** si nécessaire
5. **Documente** l'endpoint (commentaire JSDoc ou OpenAPI)

## Conventions de ce projet

Analyse les endpoints existants pour suivre les mêmes patterns :
- Structure des fichiers
- Middleware utilisé
- Format de réponse (envelope, pagination, etc.)
- Gestion des erreurs
```

**Usage** : `/create-endpoint POST /api/users`

---

## 2. Skills de workflow Git

### Commit conventionnel

**Chemin** : `~/.claude/skills/commit/SKILL.md`

```yaml
---
name: commit
description: Crée un commit avec un message conventionnel
disable-model-invocation: true
allowed-tools: Bash(git *)
---

Crée un commit en suivant la convention Conventional Commits.

## Étapes

1. Exécute `git status` et `git diff --staged` pour voir les changements
2. Si rien n'est stagé, stage les fichiers modifiés pertinents (demande confirmation)
3. Analyse les changements et détermine :
   - Le type : feat, fix, refactor, docs, test, chore, style, perf, ci, build
   - Le scope (optionnel) : le module/composant affecté
   - La description : résumé concis en impératif
4. Propose le message de commit au format : `type(scope): description`
5. Crée le commit

## Format

```
type(scope): description courte

Corps optionnel expliquant le "pourquoi" si nécessaire.

BREAKING CHANGE: description si c'est un breaking change
```

## Exemples

- `feat(auth): add OAuth2 login flow`
- `fix(api): handle null response from payment provider`
- `refactor(db): extract query builder into separate module`
```

**Usage** : `/commit`

---

### Review de PR

**Chemin** : `~/.claude/skills/review-pr/SKILL.md`

```yaml
---
name: review-pr
description: Review une pull request GitHub de manière approfondie
argument-hint: [pr-number]
disable-model-invocation: true
context: fork
agent: Explore
allowed-tools: Bash(gh *)
---

## Contexte de la PR

- Diff : !`gh pr diff $0`
- Description : !`gh pr view $0`
- Commentaires : !`gh pr view $0 --comments`
- Fichiers modifiés : !`gh pr diff $0 --name-only`

## Ta tâche

Review cette PR en analysant :

### 1. Vue d'ensemble
- Quel est l'objectif de cette PR ?
- Est-ce que les changements correspondent à la description ?

### 2. Qualité du code
- Lisibilité et maintenabilité
- Respect des conventions du projet
- Complexité inutile
- Code dupliqué

### 3. Correctness
- Bugs potentiels
- Edge cases non gérés
- Gestion d'erreurs manquante

### 4. Sécurité
- Vulnérabilités potentielles
- Données sensibles exposées
- Validation des entrées

### 5. Tests
- Couverture suffisante ?
- Cas de test manquants ?

### 6. Résumé
Donne une note globale et les actions recommandées :
- ✅ Approuver
- 🔄 Demander des changements (liste les items)
- ❓ Questions à poser à l'auteur
```

**Usage** : `/review-pr 42`

---

## 3. Skills de qualité de code

### Audit de performance

**Chemin** : `.claude/skills/perf-audit/SKILL.md`

```yaml
---
name: perf-audit
description: Analyse les performances du code et suggère des optimisations
context: fork
agent: Explore
---

Analyse les performances de $ARGUMENTS :

## Checklist d'analyse

### JavaScript/TypeScript
- [ ] Re-renders React inutiles (composants sans memo)
- [ ] Calculs coûteux sans memoization (useMemo/useCallback)
- [ ] Appels API en cascade (waterfall)
- [ ] Bundles trop gros (imports non tree-shakables)
- [ ] Boucles N+1 dans les requêtes DB

### Général
- [ ] Complexité algorithmique (O(n²) évitable ?)
- [ ] Allocations mémoire inutiles
- [ ] I/O bloquantes
- [ ] Cache manquant pour des opérations répétées
- [ ] Requêtes DB non optimisées (index manquants ?)

## Format de sortie

Pour chaque problème trouvé :
1. **Fichier et ligne** : Où se trouve le problème
2. **Impact** : Estimation de l'impact (Critique/Haut/Moyen/Bas)
3. **Problème** : Description du problème
4. **Solution** : Code corrigé ou approche recommandée
```

**Usage** : Demandez "Audite les performances de ce projet" ou `/perf-audit src/components/`

---

### Conventions de code (knowledge skill)

**Chemin** : `.claude/skills/code-style/SKILL.md`

```yaml
---
name: code-style
description: Conventions de code et patterns à suivre dans ce projet
user-invocable: false
---

## Conventions de nommage

- **Fichiers** : kebab-case (`user-profile.ts`)
- **Composants React** : PascalCase (`UserProfile.tsx`)
- **Variables/fonctions** : camelCase
- **Constantes** : UPPER_SNAKE_CASE
- **Types/Interfaces** : PascalCase, préfixe I pour les interfaces (`IUserProfile`)

## Structure des imports

Ordre des imports :
1. Modules externes (react, lodash, etc.)
2. Modules internes absolus (@/components, @/utils)
3. Modules relatifs (./utils, ../hooks)
4. Styles

## Gestion d'erreurs

- Toujours utiliser des error boundaries pour les composants React
- Utiliser des types d'erreur personnalisés (`AppError`, `ValidationError`)
- Logger les erreurs avec le contexte suffisant
- Ne jamais avaler les erreurs silencieusement

## Tests

- Fichier de test à côté du fichier source : `foo.ts` → `foo.test.ts`
- Describe blocks par fonctionnalité
- Un assert par test quand possible
- Utiliser des factories pour les données de test
```

Ce skill est invisible dans le menu `/` mais Claude le charge automatiquement quand il écrit du code pour votre projet.

---

## 4. Skills de documentation

### Documenter un module

**Chemin** : `.claude/skills/doc-module/SKILL.md`

```yaml
---
name: doc-module
description: Génère une documentation complète pour un module
argument-hint: [module-path]
disable-model-invocation: true
---

Génère une documentation complète pour le module à $ARGUMENTS.

## Structure de la documentation

### 1. Vue d'ensemble
- Objectif du module
- Où il s'inscrit dans l'architecture

### 2. API publique
Pour chaque export :
- Signature avec types
- Description
- Paramètres avec types et valeurs par défaut
- Valeur de retour
- Exemple d'utilisation

### 3. Diagramme de dépendances
ASCII art montrant :
- Ce dont le module dépend
- Ce qui dépend du module

### 4. Exemples
Au moins 2 exemples réalistes d'utilisation.

### 5. Gotchas
Points d'attention, pièges courants, edge cases.

## Format de sortie

Génère un fichier markdown dans le même répertoire que le module :
`[nom-du-module].doc.md`
```

**Usage** : `/doc-module src/services/auth/`

---

## 5. Skills avec scripts embarqués

### Analyseur de dépendances

**Chemin** : `~/.claude/skills/dep-analyzer/SKILL.md`

```yaml
---
name: dep-analyzer
description: Analyse les dépendances du projet et identifie les problèmes
allowed-tools: Bash(node *), Read, Grep, Glob
disable-model-invocation: true
---

Analyse les dépendances de ce projet.

## Étapes

1. Lis le `package.json` (ou équivalent selon le langage)
2. Identifie :
   - Dépendances non utilisées
   - Dépendances dupliquées (versions différentes)
   - Dépendances avec des vulnérabilités connues
   - Dépendances outdated
3. Exécute `npm audit` (ou équivalent) si possible
4. Génère un rapport

## Format du rapport

| Dépendance | Version actuelle | Dernière version | Statut | Action |
|:---|:---|:---|:---|:---|
| package-name | 1.0.0 | 2.0.0 | ⚠️ Majeure | Vérifier breaking changes |

### Résumé
- Total dépendances : X
- À jour : X
- Mineures disponibles : X
- Majeures disponibles : X
- Vulnérabilités : X
```

**Usage** : `/dep-analyzer`

---

## 6. Skills avec contexte dynamique

### Résumé de sprint

**Chemin** : `.claude/skills/sprint-summary/SKILL.md`

```yaml
---
name: sprint-summary
description: Résume l'avancement du sprint en cours
context: fork
agent: Explore
disable-model-invocation: true
allowed-tools: Bash(gh *), Bash(git *)
---

## Données du sprint

- Issues ouvertes : !`gh issue list --state open --limit 50`
- PRs ouvertes : !`gh pr list --state open`
- PRs mergées cette semaine : !`gh pr list --state merged --search "merged:>=$(date -d '7 days ago' +%Y-%m-%d)"`
- Commits récents : !`git log --oneline --since="7 days ago"`

## Ta tâche

Génère un résumé de sprint avec :

1. **Progression** : Nombre d'issues fermées vs ouvertes
2. **PRs** : État des PRs, celles en attente de review
3. **Highlights** : Les changements les plus significatifs
4. **Blockers** : Issues ou PRs bloquées
5. **Prochaines étapes** : Recommandations pour la suite
```

**Usage** : `/sprint-summary`

---

## 7. Subagents spécialisés

### Subagent : Debugger

**Chemin** : `~/.claude/agents/debugger.md`

```yaml
---
name: debugger
description: Expert en debugging. Utiliser quand un bug est signalé ou quand quelque chose ne fonctionne pas comme attendu.
tools: Read, Grep, Glob, Bash
model: opus
---

Tu es un expert en debugging. Quand on te signale un bug :

## Méthodologie

1. **Reproduire** : Comprends et reproduis le problème
2. **Localiser** : Utilise une approche systématique pour trouver la cause
   - Cherche les logs d'erreur
   - Trace le flux de données
   - Identifie les changements récents qui pourraient être la cause
3. **Diagnostiquer** : Identifie la cause racine (pas juste les symptômes)
4. **Expliquer** : Décris clairement :
   - Ce qui se passe
   - Pourquoi ça se passe
   - Quel est le fix recommandé
   - Comment prévenir la récurrence

## Principes

- Ne saute jamais à la solution sans comprendre la cause
- Vérifie les hypothèses avec des données
- Considère les effets de bord
- Cherche des patterns (le bug se reproduit-il ailleurs ?)
```

### Subagent : Refactoring assistant

**Chemin** : `.claude/agents/refactorer.md`

```yaml
---
name: refactorer
description: Assistant de refactoring. Utiliser pour restructurer du code sans changer le comportement.
tools: Read, Grep, Glob, Write, Edit, Bash
model: sonnet
isolation: worktree
---

Tu es un expert en refactoring. Tu travailles dans un worktree isolé
pour garantir la sécurité des changements.

## Principes

1. **Pas de changement de comportement** : Le code doit faire exactement la même chose après
2. **Tests en premier** : Vérifie que les tests passent AVANT et APRÈS le refactoring
3. **Petits pas** : Fais des changements incrémentaux, pas une réécriture totale
4. **Motifs communs** :
   - Extract method/function
   - Extract class/module
   - Rename pour clarifier l'intention
   - Simplifier les conditionnels
   - Éliminer la duplication
   - Inverser les dépendances

## Workflow

1. Exécute les tests existants
2. Identifie les refactorings à faire
3. Applique chaque refactoring un par un
4. Exécute les tests après chaque changement
5. Si un test casse, annule le dernier changement
6. Résume tous les changements effectués
```

---

## Combiner Skills et Subagents

### Exemple : Subagent avec skills préchargées

```yaml
---
name: fullstack-dev
description: Développeur fullstack qui suit les conventions du projet
tools: Read, Grep, Glob, Write, Edit, Bash
model: sonnet
skills:
  - code-style
  - api-conventions
  - testing-conventions
---

Tu es un développeur fullstack senior. Tu utilises les conventions
du projet (chargées via les skills) pour écrire du code cohérent.

Quand on te demande d'implémenter une feature :
1. Comprends les exigences
2. Planifie l'implémentation
3. Écris le code en suivant les conventions
4. Écris les tests
5. Vérifie que tout passe
```

### Exemple : Skill qui délègue à un subagent custom

```yaml
---
name: implement-feature
description: Implémente une feature complète
context: fork
agent: fullstack-dev
disable-model-invocation: true
---

Implémente la feature suivante : $ARGUMENTS

Assure-toi de :
1. Créer tous les fichiers nécessaires
2. Suivre les conventions du projet
3. Écrire des tests complets
4. Vérifier que tout compile et passe
```

**Usage** : `/implement-feature Ajouter un système de notifications par email`
