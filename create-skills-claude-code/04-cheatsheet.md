# Cheatsheet : Skills & Subagents Claude Code

Référence rapide pour créer des skills et subagents.

---

## Skill — Structure minimale

```
.claude/skills/ma-skill/SKILL.md
```

```yaml
---
name: ma-skill
description: Description pour Claude
---

Instructions ici.
```

---

## Subagent — Structure minimale

```
.claude/agents/mon-agent.md
```

```yaml
---
name: mon-agent
description: Quand déléguer à cet agent
---

Tu es un assistant spécialisé dans...
```

---

## Emplacements

### Skills

| Scope | Chemin |
|:---|:---|
| Personnel | `~/.claude/skills/<nom>/SKILL.md` |
| Projet | `.claude/skills/<nom>/SKILL.md` |

### Subagents

| Scope | Chemin |
|:---|:---|
| Personnel | `~/.claude/agents/<nom>.md` |
| Projet | `.claude/agents/<nom>.md` |

---

## Frontmatter Skill — Champs clés

```yaml
---
name: nom-skill              # Nom (défaut: nom du répertoire)
description: ...              # Quand utiliser (RECOMMANDÉ)
argument-hint: [arg1] [arg2]  # Hint d'autocomplétion
disable-model-invocation: true # Utilisateur uniquement
user-invocable: false          # Claude uniquement
allowed-tools: Read, Grep      # Outils autorisés
model: sonnet                  # Modèle à utiliser
context: fork                  # Exécuter en sous-agent
agent: Explore                 # Type de sous-agent
---
```

## Frontmatter Subagent — Champs clés

```yaml
---
name: nom-agent               # Identifiant unique (REQUIS)
description: ...               # Quand déléguer (REQUIS)
tools: Read, Grep, Glob        # Outils (défaut: tous)
disallowedTools: Write, Edit   # Outils interdits
model: sonnet                  # sonnet | opus | haiku | inherit
permissionMode: default        # default | acceptEdits | dontAsk | bypassPermissions | plan
maxTurns: 20                   # Limite de tours
skills:                        # Skills à précharger
  - skill-1
  - skill-2
memory: project                # user | project | local
background: true               # Exécution en arrière-plan
isolation: worktree             # Isolation git worktree
---
```

---

## Variables de substitution (Skills)

| Variable | Description |
|:---|:---|
| `$ARGUMENTS` | Tous les arguments |
| `$ARGUMENTS[0]` ou `$0` | Premier argument |
| `$ARGUMENTS[1]` ou `$1` | Deuxième argument |
| `${CLAUDE_SESSION_ID}` | ID de session |

---

## Contexte dynamique (Skills)

```yaml
- Diff : !`gh pr diff`
- Branch : !`git branch --show-current`
- Date : !`date +%Y-%m-%d`
```

Les commandes `` !`...` `` s'exécutent AVANT que Claude voie le contenu.

---

## Invocation

### Skills

| Méthode | Exemple |
|:---|:---|
| Directe | `/ma-skill argument` |
| Automatique | Claude la charge si pertinent |

### Subagents

Claude délègue automatiquement en fonction de la `description`.

---

## Modèles disponibles

| Alias | Usage |
|:---|:---|
| `haiku` | Rapide, peu coûteux, exploration |
| `sonnet` | Équilibré, recommandé par défaut |
| `opus` | Le plus capable, tâches complexes |
| `inherit` | Même que la conversation (défaut) |

---

## Permissions (Skills)

```
# Autoriser
Skill(nom)           # Match exact
Skill(nom *)         # Préfixe

# Refuser
Skill(deploy *)      # Bloquer deploy et ses arguments
Skill                # Bloquer toutes les skills
```

---

## Commandes utiles

```
/agents           # Gérer les subagents (interactif)
/context          # Voir le contexte chargé (vérifier les skills)
claude agents     # Lister les agents (CLI, non interactif)
```

---

## Arbre décisionnel rapide

```
Besoin d'ajouter une connaissance/convention ?
  → Skill avec user-invocable: false

Besoin d'une commande manuelle ?
  → Skill avec disable-model-invocation: true

Besoin de déléguer une tâche complexe ?
  → Subagent avec les bons outils et modèle

Besoin d'exécuter en isolation ?
  → Skill avec context: fork
  → OU Subagent avec isolation: worktree

Besoin d'un workflow en lecture seule ?
  → allowed-tools: Read, Grep, Glob
  → OU agent: Explore

Besoin de mémoire entre sessions ?
  → Subagent avec memory: user/project/local
```

---

## Erreurs courantes

| Problème | Solution |
|:---|:---|
| Skill pas détectée | Vérifier le chemin et le nom du fichier `SKILL.md` |
| Skill se déclenche trop | Rendre la `description` plus spécifique ou `disable-model-invocation: true` |
| Subagent pas chargé | Redémarrer la session ou utiliser `/agents` |
| Trop de skills, budget dépassé | Lancer `/context`, ajuster `SLASH_COMMAND_TOOL_CHAR_BUDGET` |
| `context: fork` sans résultat | Ajouter des instructions de tâche concrètes (pas juste des guidelines) |
