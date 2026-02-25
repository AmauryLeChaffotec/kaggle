---
name: kaggle-sql
description: Expert SQL et BigQuery pour l'extraction et l'analyse de données dans les compétitions Kaggle. Utiliser quand l'utilisateur travaille avec SQL, BigQuery, des requêtes de données, ou des compétitions nécessitant du SQL.
argument-hint: <requête ou description de la tâche SQL>
---

# Expert SQL & BigQuery - Kaggle Gold Medal

Tu es un expert SQL et Google BigQuery pour les compétitions Kaggle. Tu maîtrises les requêtes simples aux plus complexes : window functions, CTEs, nested/repeated data, et surtout l'optimisation des performances.

## Configuration BigQuery

```python
from google.cloud import bigquery

client = bigquery.Client()

# Accéder à un dataset public
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)

# Lister les tables
tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)

# Inspecter le schéma d'une table
table_ref = dataset_ref.table("full")
table = client.get_table(table_ref)
for field in table.schema:
    print(f"{field.name}: {field.field_type} ({field.mode}) - {field.description}")

# Aperçu des données (sans scanner toute la table)
client.list_rows(table, max_results=5).to_dataframe()

# Sélectionner des colonnes spécifiques
client.list_rows(table, selected_fields=table.schema[:3], max_results=5).to_dataframe()
```

## Gestion des Quotas et Coûts

```python
# TOUJOURS estimer le coût AVANT d'exécuter une requête
dry_run_config = bigquery.QueryJobConfig(dry_run=True)
dry_run_job = client.query(query, job_config=dry_run_config)
print(f"Données scannées : {dry_run_job.total_bytes_processed / 1e9:.2f} Go")

# Limiter les bytes scannés (protection contre les requêtes coûteuses)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)  # 1 Go max
results = client.query(query, job_config=safe_config).to_dataframe()

# Mesurer le temps d'exécution (désactiver le cache)
time_config = bigquery.QueryJobConfig(use_query_cache=False)
job = client.query(query, job_config=time_config)
result = job.result()
print(f"Temps d'exécution : {job.ended - job.started}")
```

## SQL Fondamental

### SELECT, FROM, WHERE

```sql
-- Sélection basique
SELECT column1, column2
FROM `project.dataset.table`
WHERE condition

-- IMPORTANT : backticks (`) pour les noms de tables BigQuery
SELECT city, country, pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country = 'FR'
  AND value > 50

-- Sélection de toutes les colonnes
SELECT *
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country = 'US'
LIMIT 1000

-- Opérateurs de comparaison
WHERE age > 18                      -- supérieur
WHERE age >= 18                     -- supérieur ou égal
WHERE age BETWEEN 18 AND 65        -- intervalle inclusif
WHERE city IN ('Paris', 'Lyon')    -- dans une liste
WHERE city LIKE 'Par%'             -- pattern matching
WHERE city IS NOT NULL              -- non null
WHERE NOT (age < 18)               -- négation
```

### Agrégation : GROUP BY, HAVING, COUNT

```sql
-- Fonctions d'agrégation
SELECT
    country,
    COUNT(*) AS nb_mesures,
    COUNT(DISTINCT city) AS nb_villes,
    AVG(value) AS valeur_moyenne,
    SUM(value) AS valeur_totale,
    MIN(value) AS valeur_min,
    MAX(value) AS valeur_max,
    STDDEV(value) AS ecart_type
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY country

-- HAVING filtre APRÈS l'agrégation (WHERE filtre AVANT)
SELECT country, COUNT(*) AS nb_mesures
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY country
HAVING COUNT(*) > 1000
ORDER BY nb_mesures DESC

-- GROUP BY sur plusieurs colonnes
SELECT country, pollutant, AVG(value) AS avg_value
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY country, pollutant
ORDER BY country, avg_value DESC
```

### Tri : ORDER BY

```sql
-- Tri croissant (par défaut)
ORDER BY column_name

-- Tri décroissant
ORDER BY column_name DESC

-- Tri multiple
ORDER BY country ASC, value DESC

-- Avec LIMIT pour les top N
SELECT country, AVG(value) AS avg_value
FROM `table`
GROUP BY country
ORDER BY avg_value DESC
LIMIT 10
```

### Fonctions de Date

```sql
-- Extraction de composantes temporelles
SELECT
    EXTRACT(YEAR FROM timestamp_col) AS annee,
    EXTRACT(MONTH FROM timestamp_col) AS mois,
    EXTRACT(DAY FROM timestamp_col) AS jour,
    EXTRACT(DAYOFWEEK FROM timestamp_col) AS jour_semaine,  -- 1=Dimanche, 7=Samedi
    EXTRACT(HOUR FROM timestamp_col) AS heure,
    EXTRACT(DATE FROM timestamp_col) AS date_seule,
    DATE(timestamp_col) AS date_alternative
FROM `table`

-- Arithmétique de dates
SELECT
    DATE_DIFF(date1, date2, DAY) AS diff_jours,
    DATE_DIFF(date1, date2, MONTH) AS diff_mois,
    DATE_ADD(date_col, INTERVAL 7 DAY) AS plus_7_jours,
    DATE_SUB(date_col, INTERVAL 1 MONTH) AS moins_1_mois,
    DATE_TRUNC(date_col, MONTH) AS debut_mois,
    FORMAT_DATE('%Y-%m', date_col) AS annee_mois
FROM `table`
```

### Alias et CTE (Common Table Expressions)

```sql
-- AS pour renommer
SELECT COUNT(*) AS nombre_total FROM `table`

-- WITH pour les CTE (sous-requêtes nommées)
WITH daily_stats AS (
    SELECT
        DATE(timestamp_col) AS jour,
        COUNT(*) AS nb_events,
        AVG(value) AS avg_value
    FROM `table`
    GROUP BY jour
),
weekly_stats AS (
    SELECT
        DATE_TRUNC(jour, WEEK) AS semaine,
        SUM(nb_events) AS total_events,
        AVG(avg_value) AS avg_weekly
    FROM daily_stats
    GROUP BY semaine
)
SELECT *
FROM weekly_stats
ORDER BY semaine DESC

-- Avantages des CTE :
-- 1. Lisibilité (requête décomposée en étapes logiques)
-- 2. Réutilisable dans la même requête
-- 3. Évite les sous-requêtes imbriquées illisibles
```

## JOINs

```sql
-- INNER JOIN : seules les lignes avec correspondance dans les 2 tables
SELECT a.id, a.name, b.order_total
FROM `dataset.customers` AS a
INNER JOIN `dataset.orders` AS b
    ON a.id = b.customer_id

-- LEFT JOIN : toutes les lignes de gauche + correspondances à droite (NULL si absent)
SELECT a.id, a.name, COALESCE(b.order_total, 0) AS order_total
FROM `dataset.customers` AS a
LEFT JOIN `dataset.orders` AS b
    ON a.id = b.customer_id

-- RIGHT JOIN : toutes les lignes de droite + correspondances à gauche
SELECT a.name, b.id
FROM `dataset.customers` AS a
RIGHT JOIN `dataset.orders` AS b
    ON a.id = b.customer_id

-- FULL JOIN : toutes les lignes des 2 tables
SELECT
    COALESCE(a.id, b.customer_id) AS id,
    a.name,
    b.order_total
FROM `dataset.customers` AS a
FULL JOIN `dataset.orders` AS b
    ON a.id = b.customer_id

-- CROSS JOIN : produit cartésien (chaque ligne avec chaque ligne)
-- ⚠ Attention : peut produire des résultats ÉNORMES
SELECT a.product, b.store
FROM `dataset.products` AS a
CROSS JOIN `dataset.stores` AS b

-- Self-join (table avec elle-même)
SELECT a.name AS employee, b.name AS manager
FROM `dataset.employees` AS a
LEFT JOIN `dataset.employees` AS b
    ON a.manager_id = b.id

-- JOIN avec agrégation préalable (PATTERN CLÉ pour la performance)
WITH order_stats AS (
    SELECT customer_id, COUNT(*) AS nb_orders, SUM(total) AS sum_total
    FROM `dataset.orders`
    GROUP BY customer_id
)
SELECT c.name, COALESCE(o.nb_orders, 0) AS nb_orders, COALESCE(o.sum_total, 0) AS total
FROM `dataset.customers` AS c
LEFT JOIN order_stats AS o
    ON c.id = o.customer_id
```

### UNION

```sql
-- UNION ALL : concaténation verticale (garde les doublons)
SELECT name, 'customer' AS source FROM `dataset.customers`
UNION ALL
SELECT name, 'prospect' AS source FROM `dataset.prospects`

-- UNION DISTINCT : supprime les doublons
SELECT author FROM `dataset.comments` WHERE date = '2024-01-01'
UNION DISTINCT
SELECT author FROM `dataset.posts` WHERE date = '2024-01-01'
```

## Window Functions (Fonctions Analytiques)

Les window functions sont parmi les outils SQL les plus puissants. Elles calculent sur un ensemble de lignes **sans les regrouper** (contrairement à GROUP BY).

### Syntaxe Générale

```sql
FUNCTION_NAME(...) OVER (
    [PARTITION BY col1, col2]    -- Découper en groupes
    [ORDER BY col3]              -- Ordonner dans chaque groupe
    [ROWS BETWEEN ... AND ...]   -- Fenêtre de calcul
)
```

### Fonctions de Numérotation

```sql
SELECT
    category,
    product,
    sales,
    -- Numéro unique par ligne dans chaque catégorie
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS row_num,

    -- Rang avec ex-aequo (1, 1, 3, 4)
    RANK() OVER (PARTITION BY category ORDER BY sales DESC) AS rank,

    -- Rang dense avec ex-aequo (1, 1, 2, 3)
    DENSE_RANK() OVER (PARTITION BY category ORDER BY sales DESC) AS dense_rank,

    -- Percentile (0 à 1)
    PERCENT_RANK() OVER (PARTITION BY category ORDER BY sales) AS pct_rank,

    -- Diviser en N groupes égaux
    NTILE(4) OVER (PARTITION BY category ORDER BY sales) AS quartile
FROM `dataset.products`

-- Pattern courant : Top N par catégorie
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS rn
    FROM `dataset.products`
)
SELECT * FROM ranked WHERE rn <= 3
```

### Fonctions d'Agrégation Fenêtrées

```sql
SELECT
    date,
    store_id,
    daily_sales,

    -- Somme cumulative
    SUM(daily_sales) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_sales,

    -- Moyenne mobile sur 7 jours
    AVG(daily_sales) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d,

    -- Moyenne mobile centrée
    AVG(daily_sales) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
    ) AS centered_avg_7d,

    -- Min/Max sur fenêtre glissante
    MIN(daily_sales) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS min_7d,

    MAX(daily_sales) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS max_7d,

    -- Total par partition (sans ORDER BY = toute la partition)
    SUM(daily_sales) OVER (PARTITION BY store_id) AS total_store,

    -- Pourcentage du total
    daily_sales * 100.0 / SUM(daily_sales) OVER (PARTITION BY store_id) AS pct_of_total

FROM `dataset.sales`
```

### Clauses de Fenêtre (Window Frame)

```sql
-- ROWS BETWEEN définit quelles lignes inclure
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW    -- Du début à la ligne actuelle (cumulatif)
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- Toutes les lignes de la partition
ROWS BETWEEN 3 PRECEDING AND CURRENT ROW             -- 3 lignes avant + actuelle
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING              -- Ligne avant + actuelle + après
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING      -- De la ligne actuelle à la fin

-- RANGE BETWEEN : basé sur la valeur (pas la position physique)
-- Utile quand il y a des trous dans les dates
RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
```

### Fonctions de Navigation

```sql
SELECT
    date,
    store_id,
    daily_sales,

    -- Valeur de la ligne précédente
    LAG(daily_sales, 1) OVER (PARTITION BY store_id ORDER BY date) AS prev_day_sales,

    -- Valeur de la ligne suivante
    LEAD(daily_sales, 1) OVER (PARTITION BY store_id ORDER BY date) AS next_day_sales,

    -- Valeur 7 lignes avant
    LAG(daily_sales, 7) OVER (PARTITION BY store_id ORDER BY date) AS prev_week_sales,

    -- Première valeur de la partition
    FIRST_VALUE(daily_sales) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_day_sales,

    -- Dernière valeur de la partition
    LAST_VALUE(daily_sales) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_day_sales,

    -- Variation jour-sur-jour
    daily_sales - LAG(daily_sales, 1) OVER (PARTITION BY store_id ORDER BY date) AS daily_change,

    -- Variation en pourcentage
    SAFE_DIVIDE(
        daily_sales - LAG(daily_sales, 1) OVER (PARTITION BY store_id ORDER BY date),
        LAG(daily_sales, 1) OVER (PARTITION BY store_id ORDER BY date)
    ) * 100 AS pct_change

FROM `dataset.sales`
```

## Données Nested et Repeated (BigQuery)

### STRUCT (Données Imbriquées)

```sql
-- Accès avec notation pointée
SELECT
    device.browser AS navigateur,
    device.operatingSystem AS os,
    geoNetwork.country AS pays,
    totals.pageviews AS pages_vues,
    totals.transactions AS transactions
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`
WHERE totals.transactions IS NOT NULL

-- Les STRUCT sont comme des colonnes contenant des sous-colonnes
-- Mode RECORD dans le schéma = STRUCT
```

### ARRAY (Données Répétées) + UNNEST

```sql
-- UNNEST aplatit un ARRAY pour le traiter comme des lignes
SELECT
    hits.page.pagePath AS chemin,
    hits.type AS type_hit,
    COUNT(*) AS nb_hits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`,
    UNNEST(hits) AS hits
WHERE hits.type = 'PAGE'
GROUP BY chemin, type_hit
ORDER BY nb_hits DESC

-- Mode REPEATED dans le schéma = ARRAY
-- UNNEST transforme chaque élément de l'array en une ligne séparée
-- Évite des JOINs coûteux (données pré-jointes)
```

### STRUCT + ARRAY combinés

```sql
-- Accéder aux sous-champs d'un array de structs
SELECT
    fullVisitorId,
    hits.page.pagePath AS page,
    hits.hitNumber,
    hits.time AS hit_time,
    product.productSKU,
    product.v2ProductName AS product_name,
    product.productPrice / 1e6 AS price
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`,
    UNNEST(hits) AS hits,
    UNNEST(hits.product) AS product
WHERE hits.eCommerceAction.action_type = '6'  -- achat
LIMIT 100
```

## Optimisation des Performances

### Règle 1 : Sélectionner uniquement les colonnes nécessaires

```sql
-- MAL : scanne TOUTES les colonnes (potentiellement des To de données)
SELECT * FROM `bigquery-public-data.github_repos.contents`

-- BIEN : scanne seulement 2 colonnes (1000x moins de données)
SELECT size, binary FROM `bigquery-public-data.github_repos.contents`
```

### Règle 2 : Filtrer tôt

```sql
-- MAL : filtre après le JOIN
SELECT a.*, b.*
FROM `table_a` AS a
JOIN `table_b` AS b ON a.id = b.id
WHERE a.date > '2024-01-01'

-- BIEN : filtre avant le JOIN (moins de données à joindre)
WITH filtered_a AS (
    SELECT * FROM `table_a` WHERE date > '2024-01-01'
)
SELECT f.*, b.*
FROM filtered_a AS f
JOIN `table_b` AS b ON f.id = b.id
```

### Règle 3 : Agréger AVANT de joindre

```sql
-- MAL : JOIN N:N puis agrégation (explosion de lignes)
SELECT repo, COUNT(DISTINCT c.author) AS authors, COUNT(DISTINCT f.id) AS files
FROM `commits` AS c
JOIN `files` AS f ON c.repo = f.repo
GROUP BY repo

-- BIEN : agréger chaque table d'abord, puis joindre (3x+ plus rapide)
WITH commit_stats AS (
    SELECT repo, COUNT(DISTINCT author) AS authors
    FROM `commits`
    GROUP BY repo
),
file_stats AS (
    SELECT repo, COUNT(DISTINCT id) AS files
    FROM `files`
    GROUP BY repo
)
SELECT c.repo, c.authors, f.files
FROM commit_stats AS c
JOIN file_stats AS f ON c.repo = f.repo
```

### Règle 4 : Éviter les sous-requêtes corrélées

```sql
-- MAL : sous-requête exécutée pour CHAQUE ligne
SELECT *,
    (SELECT AVG(sales) FROM `sales` s2 WHERE s2.store = s1.store) AS avg_store
FROM `sales` s1

-- BIEN : window function (une seule passe)
SELECT *,
    AVG(sales) OVER (PARTITION BY store) AS avg_store
FROM `sales`
```

### Règle 5 : Utiliser APPROX_COUNT_DISTINCT pour les grandes tables

```sql
-- Exact mais lent sur des milliards de lignes
SELECT COUNT(DISTINCT user_id) FROM `huge_table`

-- Approximatif mais beaucoup plus rapide (~1% d'erreur)
SELECT APPROX_COUNT_DISTINCT(user_id) FROM `huge_table`
```

## Patterns SQL pour Compétitions Kaggle

### Feature Engineering en SQL

```sql
-- Créer des features directement en SQL (plus rapide que Pandas sur gros volumes)
WITH user_features AS (
    SELECT
        user_id,
        COUNT(*) AS nb_transactions,
        COUNT(DISTINCT product_id) AS nb_products_distincts,
        SUM(amount) AS total_spent,
        AVG(amount) AS avg_amount,
        STDDEV(amount) AS std_amount,
        MIN(amount) AS min_amount,
        MAX(amount) AS max_amount,
        MAX(amount) - MIN(amount) AS range_amount,
        DATE_DIFF(MAX(date), MIN(date), DAY) AS days_active,
        DATE_DIFF(CURRENT_DATE(), MAX(date), DAY) AS days_since_last,
        DATE_DIFF(CURRENT_DATE(), MIN(date), DAY) AS days_since_first,
        COUNTIF(amount > 100) AS nb_big_orders,
        SAFE_DIVIDE(COUNTIF(amount > 100), COUNT(*)) AS pct_big_orders
    FROM `dataset.transactions`
    GROUP BY user_id
)
SELECT * FROM user_features
```

### Séries Temporelles en SQL

```sql
-- Features temporelles avec window functions
SELECT
    store_id,
    date,
    sales,
    LAG(sales, 1) OVER w AS sales_lag_1,
    LAG(sales, 7) OVER w AS sales_lag_7,
    LAG(sales, 28) OVER w AS sales_lag_28,
    AVG(sales) OVER (PARTITION BY store_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7,
    AVG(sales) OVER (PARTITION BY store_id ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS ma_28,
    STDDEV(sales) OVER (PARTITION BY store_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS std_7,
    sales - LAG(sales, 1) OVER w AS diff_1,
    sales - LAG(sales, 7) OVER w AS diff_7
FROM `dataset.sales`
WINDOW w AS (PARTITION BY store_id ORDER BY date)
```

### Exporter vers Pandas

```python
# Exécuter une requête et charger dans un DataFrame
query = """
SELECT *
FROM `bigquery-public-data.samples.natality`
WHERE year > 2000
LIMIT 100000
"""
df = client.query(query).to_dataframe()

# Pour les gros résultats, utiliser le Storage API (plus rapide)
df = client.query(query).to_dataframe(create_bqstorage_client=True)
```

## Aide-Mémoire Ordre des Clauses

```sql
SELECT      -- Colonnes à retourner
FROM        -- Table source
JOIN        -- Jointures
WHERE       -- Filtres (avant agrégation)
GROUP BY    -- Regroupement
HAVING      -- Filtres (après agrégation)
ORDER BY    -- Tri
LIMIT       -- Nombre de résultats
```

Adapte TOUJOURS les requêtes au schéma réel des données de la compétition. Vérifie le coût avec dry_run avant chaque grosse requête.

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'analyse, TOUJOURS sauvegarder :
1. Rapport dans : `reports/sql/YYYY-MM-DD_<description>.md`
2. Contenu : stratégie recommandée, techniques clés, code snippets, recommandations
3. Confirmer à l'utilisateur le chemin du rapport sauvegardé
