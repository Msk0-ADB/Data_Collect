# ml_engine.py
# Ce fichier contient TOUS les algorithmes de Machine Learning.
# Il est le cœur analytique de l'application.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =====================================================================
# IMPORTS SCIKIT-LEARN
# Chaque import correspond à un concept ML distinct
# =====================================================================
from sklearn.linear_model import LinearRegression       # Concept 1 & 2
from sklearn.preprocessing import StandardScaler        # Normalisation des données
from sklearn.decomposition import PCA                   # Concept 3 : Réduction de dimensionnalité
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Concept 3 (LDA)
from sklearn.svm import SVC                             # Concept 4 : Classification supervisée (SVM)
from sklearn.neighbors import KNeighborsClassifier      # Concept 4 : KNN
from sklearn.tree import DecisionTreeClassifier         # Concept 4 : Arbre de décision
from sklearn.cluster import KMeans                      # Concept 5 : Classification non-supervisée
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')


class MoteurML:
    """
    Classe principale du moteur ML.
    Regroupe toutes les méthodes d'analyse et de prédiction.
    """
    
    def __init__(self, donnees: pd.DataFrame):
        """
        Initialisation avec un DataFrame pandas contenant les ventes.
        Colonnes attendues : date, recette_journaliere, quantite_totale, 
                             nb_commandes, article_le_plus_vendu
        """
        self.df = donnees.copy()
        self.scaler = StandardScaler()
    
    # =================================================================
    # CONCEPT 1 : RÉGRESSION LINÉAIRE SIMPLE
    # On prédit la recette future à partir d'une seule variable : le temps
    # =================================================================
    def regression_lineaire_simple(self):
        """
        Régression linéaire simple : Y (recette) = a*X (numéro du jour) + b
        Retourne : tendance, prédictions sur 7 jours, équation de la droite
        """
        if len(self.df) < 3:
            return {'erreur': 'Pas assez de données (minimum 3 jours)'}
        
        # X = numéro du jour (0, 1, 2, ...), Y = recette journalière
        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['recette_journaliere'].values
        
        # Entraînement du modèle
        modele = LinearRegression()
        modele.fit(X, y)
        
        # Prédictions pour les 7 prochains jours
        X_futur = np.arange(len(self.df), len(self.df) + 7).reshape(-1, 1)
        predictions = modele.predict(X_futur)
        
        # Calcul du score R² (qualité du modèle, entre 0 et 1)
        r2 = modele.score(X, y)
        
        # Détermination de la tendance
        pente = modele.coef_[0]
        if pente > 5:
            tendance = 'hausse'
        elif pente < -5:
            tendance = 'baisse'
        else:
            tendance = 'stable'
        
        return {
            'tendance': tendance,
            'pente': round(pente, 2),
            'intercept': round(modele.intercept_, 2),
            'r2_score': round(r2, 3),
            'predictions_7j': [round(p, 2) for p in predictions],
            'equation': f"Recette = {round(pente,2)} × jour + {round(modele.intercept_,2)}",
            'valeurs_reelles': y.tolist(),
            'valeurs_predites': modele.predict(X).tolist()
        }
    
    # =================================================================
    # CONCEPT 2 : RÉGRESSION LINÉAIRE MULTIPLE
    # On prédit la recette en tenant compte de plusieurs variables :
    # quantité vendue, nb commandes, jour de la semaine, etc.
    # =================================================================
    def regression_lineaire_multiple(self):
        """
        Régression multiple : Recette = a1*quantité + a2*nb_commandes + a3*jour_semaine + b
        Permet de comprendre QUELS facteurs influencent le plus les recettes.
        """
        if len(self.df) < 5:
            return {'erreur': 'Pas assez de données (minimum 5 jours)'}
        
        df_temp = self.df.copy()
        
        # Création des variables explicatives (features)
        df_temp['jour_semaine'] = pd.to_datetime(df_temp['date']).dt.dayofweek  # 0=Lundi, 6=Dimanche
        df_temp['semaine_annee'] = pd.to_datetime(df_temp['date']).dt.isocalendar().week.astype(int)
        
        # Sélection des variables X (indépendantes) et y (dépendante)
        features = ['quantite_totale', 'nb_commandes', 'jour_semaine', 'semaine_annee']
        features_disponibles = [f for f in features if f in df_temp.columns]
        
        X = df_temp[features_disponibles].fillna(0).values
        y = df_temp['recette_journaliere'].values
        
        # Normalisation des données (important pour la régression multiple)
        X_normalise = self.scaler.fit_transform(X)
        
        # Entraînement
        modele = LinearRegression()
        modele.fit(X_normalise, y)
        
        # Importance de chaque variable (valeur absolue des coefficients normalisés)
        importances = dict(zip(features_disponibles, 
                               [round(abs(c), 3) for c in modele.coef_]))
        
        # Variable la plus influente
        variable_principale = max(importances, key=importances.get)
        
        r2 = modele.score(X_normalise, y)
        
        return {
            'r2_score': round(r2, 3),
            'coefficients': {k: round(v, 3) for k, v in zip(features_disponibles, modele.coef_)},
            'importances': importances,
            'variable_principale': variable_principale,
            'interpretation': f"La variable '{variable_principale}' influence le plus les recettes",
            'prevision_demain': round(modele.predict(X_normalise[-1:].reshape(1, -1))[0] * 1.05, 2)
        }
    
    # =================================================================
    # CONCEPT 3 : RÉDUCTION DE DIMENSIONNALITÉ (PCA + LDA)
    # Permet de visualiser les données en 2D et de trouver des patterns
    # =================================================================
    def reduction_dimensionnalite(self):
        """
        PCA (Principal Component Analysis) : compresse l'information en 2 composantes.
        LDA : cherche à maximiser la séparation entre les classes (jours bons/mauvais).
        """
        if len(self.df) < 6:
            return {'erreur': 'Pas assez de données (minimum 6 jours)'}
        
        df_temp = self.df.copy()
        df_temp['jour_semaine'] = pd.to_datetime(df_temp['date']).dt.dayofweek
        
        features = ['quantite_totale', 'nb_commandes', 'jour_semaine']
        features_dispo = [f for f in features if f in df_temp.columns]
        
        X = df_temp[features_dispo].fillna(0).values
        X_normalise = StandardScaler().fit_transform(X)
        
        # --- PCA ---
        n_components = min(2, X_normalise.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_normalise)
        
        variance_expliquee = pca.explained_variance_ratio_.tolist()
        
        # --- Classification des jours (bon/moyen/mauvais) pour LDA ---
        mediane = df_temp['recette_journaliere'].median()
        q75 = df_temp['recette_journaliere'].quantile(0.75)
        
        def classer_jour(recette):
            if recette >= q75: return 2      # Bon jour
            elif recette >= mediane: return 1 # Jour moyen
            else: return 0                    # Mauvais jour
        
        y_classes = df_temp['recette_journaliere'].apply(classer_jour).values
        classes_uniques = np.unique(y_classes)
        
        # --- LDA (si au moins 2 classes différentes) ---
        resultats_lda = {}
        if len(classes_uniques) >= 2:
            try:
                lda = LinearDiscriminantAnalysis(n_components=1)
                X_lda = lda.fit_transform(X_normalise, y_classes)
                resultats_lda = {
                    'scores': X_lda.flatten().tolist(),
                    'classes': y_classes.tolist()
                }
            except Exception:
                resultats_lda = {'erreur': 'LDA non applicable avec ces données'}
        
        return {
            'pca': {
                'composantes': X_pca.tolist(),
                'variance_expliquee': [round(v*100, 1) for v in variance_expliquee],
                'variance_totale': round(sum(variance_expliquee)*100, 1)
            },
            'lda': resultats_lda,
            'classes_jours': y_classes.tolist(),
            'labels_classes': {0: 'Mauvais jour', 1: 'Jour moyen', 2: 'Bon jour'}
        }
    
    # =================================================================
    # CONCEPT 4 : CLASSIFICATION SUPERVISÉE (SVM, KNN, Arbre de décision)
    # On apprend à partir des données passées pour classifier les jours futurs
    # =================================================================
    def classification_supervisee(self):
        """
        Entraîne 3 classifieurs pour prédire si demain sera un bon/mauvais jour.
        Compare leurs performances et retourne le meilleur.
        """
        if len(self.df) < 8:
            return {'erreur': 'Pas assez de données (minimum 8 jours)'}
        
        df_temp = self.df.copy()
        df_temp['jour_semaine'] = pd.to_datetime(df_temp['date']).dt.dayofweek
        
        features = ['quantite_totale', 'nb_commandes', 'jour_semaine']
        features_dispo = [f for f in features if f in df_temp.columns]
        
        X = df_temp[features_dispo].fillna(0).values
        
        # Étiquettes : bon jour (1) ou mauvais jour (0) selon la médiane
        mediane = df_temp['recette_journaliere'].median()
        y = (df_temp['recette_journaliere'] >= mediane).astype(int).values
        
        # Normalisation
        scaler = StandardScaler()
        X_normalise = scaler.fit_transform(X)
        
        # Division train/test (80% entraînement, 20% test)
        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_normalise, y, test_size=0.2, random_state=42
            )
        else:
            # Trop peu de données : on utilise tout pour l'entraînement
            X_train, X_test = X_normalise, X_normalise
            y_train, y_test = y, y
        
        resultats = {}
        
        # --- SVM (Support Vector Machine) ---
        try:
            svm = SVC(kernel='rbf', probability=True, random_state=42)
            svm.fit(X_train, y_train)
            acc_svm = accuracy_score(y_test, svm.predict(X_test))
            pred_demain_svm = svm.predict(X_normalise[-1:].reshape(1, -1))[0]
            resultats['SVM'] = {'accuracy': round(acc_svm, 3), 'prediction_demain': int(pred_demain_svm)}
        except Exception as e:
            resultats['SVM'] = {'erreur': str(e)}
        
        # --- KNN (K plus proches voisins) ---
        try:
            knn = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
            knn.fit(X_train, y_train)
            acc_knn = accuracy_score(y_test, knn.predict(X_test))
            pred_demain_knn = knn.predict(X_normalise[-1:].reshape(1, -1))[0]
            resultats['KNN'] = {'accuracy': round(acc_knn, 3), 'prediction_demain': int(pred_demain_knn)}
        except Exception as e:
            resultats['KNN'] = {'erreur': str(e)}
        
        # --- Arbre de décision ---
        try:
            dt = DecisionTreeClassifier(max_depth=3, random_state=42)
            dt.fit(X_train, y_train)
            acc_dt = accuracy_score(y_test, dt.predict(X_test))
            pred_demain_dt = dt.predict(X_normalise[-1:].reshape(1, -1))[0]
            resultats['Arbre_Decision'] = {'accuracy': round(acc_dt, 3), 'prediction_demain': int(pred_demain_dt)}
        except Exception as e:
            resultats['Arbre_Decision'] = {'erreur': str(e)}
        
        # Meilleur modèle
        meilleur = max(
            {k: v for k, v in resultats.items() if 'accuracy' in v},
            key=lambda k: resultats[k].get('accuracy', 0),
            default='KNN'
        )
        
        return {
            'modeles': resultats,
            'meilleur_modele': meilleur,
            'prediction_demain': resultats.get(meilleur, {}).get('prediction_demain', 0),
            'interpretation': 'Bon jour probable' if resultats.get(meilleur, {}).get('prediction_demain', 0) == 1 else 'Jour difficile prévu'
        }
    
    # =================================================================
    # CONCEPT 5 : CLASSIFICATION NON-SUPERVISÉE (K-MEANS CLUSTERING)
    # Regroupe les jours similaires sans étiquettes prédéfinies
    # =================================================================
    def clustering_kmeans(self):
        """
        K-Means : regroupe automatiquement les jours en clusters (profils de vente).
        Identifie : jours de forte activité, jours calmes, jours moyens.
        """
        if len(self.df) < 4:
            return {'erreur': 'Pas assez de données (minimum 4 jours)'}
        
        df_temp = self.df.copy()
        df_temp['jour_semaine'] = pd.to_datetime(df_temp['date']).dt.dayofweek
        
        features = ['recette_journaliere', 'quantite_totale', 'nb_commandes']
        features_dispo = [f for f in features if f in df_temp.columns]
        
        X = df_temp[features_dispo].fillna(0).values
        X_normalise = StandardScaler().fit_transform(X)
        
        # Nombre de clusters (max 3 si peu de données)
        k = min(3, len(X) - 1)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_normalise)
        
        # Score de silhouette (qualité du clustering, entre -1 et 1)
        silhouette = silhouette_score(X_normalise, labels) if k > 1 else 0
        
        # Caractérisation de chaque cluster
        df_temp['cluster'] = labels
        profils = {}
        for i in range(k):
            subset = df_temp[df_temp['cluster'] == i]
            recette_moy = subset['recette_journaliere'].mean()
            
            if recette_moy == df_temp.groupby('cluster')['recette_journaliere'].mean().max():
                nom = 'Jours de forte activité'
            elif recette_moy == df_temp.groupby('cluster')['recette_journaliere'].mean().min():
                nom = 'Jours calmes'
            else:
                nom = 'Jours modérés'
            
            profils[f'Cluster_{i}'] = {
                'nom': nom,
                'nb_jours': int(len(subset)),
                'recette_moyenne': round(recette_moy, 2),
                'jours': subset['date'].astype(str).tolist()
            }
        
        return {
            'k': k,
            'labels': labels.tolist(),
            'silhouette_score': round(silhouette, 3),
            'profils': profils,
            'interpretation': f"{k} profils de jours identifiés automatiquement"
        }
    
    # =================================================================
    # ANALYSE HEBDOMADAIRE COMPLÈTE
    # =================================================================
    def analyse_hebdomadaire(self):
        """
        Analyse l'évolution semaine par semaine.
        Retourne : recette par semaine, tendance, variation en %
        """
        if len(self.df) < 2:
            return {}
        
        df_temp = self.df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['semaine'] = df_temp['date'].dt.isocalendar().week.astype(int)
        df_temp['annee'] = df_temp['date'].dt.year
        
        hebdo = df_temp.groupby(['annee', 'semaine']).agg(
            recette_totale=('recette_journaliere', 'sum'),
            nb_jours=('recette_journaliere', 'count'),
            recette_moyenne=('recette_journaliere', 'mean')
        ).reset_index()
        
        # Variation semaine par semaine
        hebdo['variation_pct'] = hebdo['recette_totale'].pct_change() * 100
        hebdo['evolution'] = hebdo['variation_pct'].apply(
            lambda x: 'hausse' if x > 2 else ('baisse' if x < -2 else 'stable')
        )
        
        return hebdo.fillna(0).to_dict(orient='records')
