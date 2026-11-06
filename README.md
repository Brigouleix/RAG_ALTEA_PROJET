# Assistant RAG Altéa Services

Prototype d'assistant qui répond aux questions des salariés à partir de la
documentation interne, en citant ses sources.

## Installation
    uv sync
    ollama pull llama3.2

## Ordre de lancement
1. notebooks/1_collecte.ipynb
2. notebooks/2_nettoyage.ipynb
3. notebooks/3_decoupage.ipynb
4. notebooks/4_indexation.ipynb
5. uv run streamlit run app.py

## Exemple
Question : Combien de jours de télétravail par semaine ?
Réponse : jusqu'à 2 jours par semaine, sauf le mardi (teletravail.pdf, page 1).

## Résultats de l'évaluation
- Recherche : 100 % (10 questions, k=3), MRR 0,95
- Réponses justes : XX % | correctement sourcées : XX %
- Amélioration testée : prompt (règle sur les délais), de XX % à XX %

## Limites connues
- Découpage au caractère (mots parfois coupés)
- Modèle local de petite taille : erreurs possibles sur les nuances
- Jeu de test de 10 questions, proches du vocabulaire des documents

## Auteurs
Antoine Brigouleix