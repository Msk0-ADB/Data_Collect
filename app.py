# app.py
# Fichier principal de l'application Flask.
# Il définit toutes les routes (URLs) et orchestre les autres modules.

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

from models import db, Article, Commande, LigneCommande
from ml_engine import MoteurML
from gemini_service import ServiceGemini

load_dotenv()

# =====================================================================
# CONFIGURATION DE L'APPLICATION
# =====================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_secret_key_123')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ventes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialisation de la base de données
db.init_app(app)

# Création des tables au démarrage (important pour Render)
with app.app_context():
    db.create_all()

# Initialisation du service Gemini
gemini = ServiceGemini()


def get_dataframe_ventes():
    """
    Extrait les données de la base SQLite et les convertit en DataFrame pandas.
    C'est le format requis par le moteur ML.
    """
    commandes = Commande.query.order_by(Commande.date).all()
    
    if not commandes:
        return pd.DataFrame(columns=['date', 'recette_journaliere', 'quantite_totale', 'nb_commandes'])
    
    # Agrégation par jour
    donnees = {}
    for cmd in commandes:
        d = str(cmd.date)
        if d not in donnees:
            donnees[d] = {'date': d, 'recette_journaliere': 0, 'quantite_totale': 0, 'nb_commandes': 0}
        donnees[d]['recette_journaliere'] += cmd.total
        donnees[d]['quantite_totale'] += sum(l.quantite for l in cmd.lignes)
        donnees[d]['nb_commandes'] += 1
    
    return pd.DataFrame(list(donnees.values()))


# =====================================================================
# ROUTES PRINCIPALES
# =====================================================================

@app.route('/')
def index():
    """Page d'accueil : tableau de bord avec statistiques générales."""
    
    # Statistiques de base
    total_commandes = Commande.query.count()
    recette_totale = db.session.query(db.func.sum(Commande.total)).scalar() or 0
    nb_articles = Article.query.count()
    
    # Ventes d'aujourd'hui
    aujourd_hui = date.today()
    commandes_today = Commande.query.filter_by(date=aujourd_hui).all()
    recette_today = sum(c.total for c in commandes_today)
    
    # Dernières commandes (10)
    dernieres_commandes = Commande.query.order_by(Commande.date_creation.desc()).limit(10).all()
    
    # Données pour les graphiques (30 derniers jours)
    df = get_dataframe_ventes()
    graphique_data = []
    if not df.empty:
        graphique_data = df.tail(30).to_dict(orient='records')
    
    # Analyse ML rapide si données suffisantes
    analyse_rapide = {}
    if len(df) >= 3:
        ml = MoteurML(df)
        rl = ml.regression_lineaire_simple()
        analyse_rapide = {
            'tendance': rl.get('tendance', 'stable'),
            'pente': rl.get('pente', 0),
            'r2': rl.get('r2_score', 0)
        }
    
    return render_template('index.html',
        total_commandes=total_commandes,
        recette_totale=round(recette_totale, 2),
        nb_articles=nb_articles,
        recette_today=round(recette_today, 2),
        dernieres_commandes=dernieres_commandes,
        graphique_data=graphique_data,
        analyse_rapide=analyse_rapide
    )


@app.route('/saisie', methods=['GET', 'POST'])
def saisie():
    """Page de saisie d'une nouvelle commande."""
    articles = Article.query.all()
    
    if request.method == 'POST':
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'Données manquantes'}), 400
        
        try:
            # Création de la commande
            commande = Commande(
                date=datetime.strptime(data.get('date', str(date.today())), '%Y-%m-%d').date(),
                client=data.get('client', 'Client anonyme')
            )
            db.session.add(commande)
            db.session.flush()  # Obtenir l'ID de la commande
            
            # Ajout des lignes de commande
            for ligne_data in data.get('lignes', []):
                article = Article.query.get(ligne_data['article_id'])
                if article and ligne_data.get('quantite', 0) > 0:
                    ligne = LigneCommande(
                        commande_id=commande.id,
                        article_id=article.id,
                        quantite=int(ligne_data['quantite']),
                        prix_unitaire=article.prix_unitaire
                    )
                    db.session.add(ligne)
            
            db.session.flush()
            commande.calculer_total()
            db.session.commit()
            
            return jsonify({'success': True, 'message': f'Commande #{commande.id} enregistrée', 'total': commande.total})
        
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': str(e)}), 500
    
    return render_template('saisie.html', articles=articles)


@app.route('/articles', methods=['GET', 'POST'])
def articles():
    """Gestion des articles (liste + ajout)."""
    if request.method == 'POST':
        data = request.get_json()
        try:
            article = Article(
                nom=data['nom'],
                prix_unitaire=float(data['prix']),
                categorie=data.get('categorie', 'Général')
            )
            db.session.add(article)
            db.session.commit()
            return jsonify({'success': True, 'id': article.id, 'message': 'Article ajouté'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': str(e)}), 500
    
    tous_articles = Article.query.all()
    return render_template('saisie.html', articles=tous_articles, mode='articles')


@app.route('/analyse')
def analyse():
    """Page d'analyse ML complète."""
    df = get_dataframe_ventes()
    
    resultats_ml = {}
    if len(df) >= 3:
        ml = MoteurML(df)
        resultats_ml['regression_simple'] = ml.regression_lineaire_simple()
        
        if len(df) >= 5:
            resultats_ml['regression_multiple'] = ml.regression_lineaire_multiple()
        
        if len(df) >= 6:
            resultats_ml['reduction_dim'] = ml.reduction_dimensionnalite()
        
        if len(df) >= 8:
            resultats_ml['classification'] = ml.classification_supervisee()
        
        if len(df) >= 4:
            resultats_ml['clustering'] = ml.clustering_kmeans()
        
        resultats_ml['hebdomadaire'] = ml.analyse_hebdomadaire()
    
    return render_template('analyse.html', resultats=resultats_ml, nb_jours=len(df))


# =====================================================================
# ROUTES API (JSON) — utilisées par le JavaScript du frontend
# =====================================================================

@app.route('/api/analyse-ia', methods=['POST'])
def analyse_ia():
    """
    Lance l'analyse Gemini AI avec les données actuelles.
    Retourne des conseils personnalisés en JSON.
    """
    df = get_dataframe_ventes()
    
    if df.empty:
        return jsonify({'erreur': 'Aucune donnée disponible. Saisissez des ventes d\'abord.'})
    
    # Résumé des données pour Gemini
    resume = {
        'nb_jours_analyses': len(df),
        'recette_totale': round(df['recette_journaliere'].sum(), 2),
        'recette_moyenne_jour': round(df['recette_journaliere'].mean(), 2),
        'recette_max': round(df['recette_journaliere'].max(), 2),
        'recette_min': round(df['recette_journaliere'].min(), 2),
        'quantite_totale_vendue': int(df['quantite_totale'].sum()),
        'derniere_semaine': df.tail(7)['recette_journaliere'].tolist(),
        'tendance_recente': 'hausse' if len(df) >= 2 and df.iloc[-1]['recette_journaliere'] > df.iloc[-2]['recette_journaliere'] else 'baisse'
    }
    
    # Top articles
    lignes = LigneCommande.query.all()
    if lignes:
        ventes_par_article = {}
        for l in lignes:
            nom = l.article.nom if l.article else 'Inconnu'
            ventes_par_article[nom] = ventes_par_article.get(nom, 0) + l.quantite
        top = sorted(ventes_par_article.items(), key=lambda x: x[1], reverse=True)[:5]
        resume['top_articles'] = [{'article': a, 'quantite': q} for a, q in top]
    
    conseils = gemini.analyser_ventes(resume)
    return jsonify(conseils)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat interactif avec Gemini."""
    data = request.get_json()
    message = data.get('message', '')
    
    df = get_dataframe_ventes()
    contexte = {
        'nb_jours': len(df),
        'recette_totale': round(df['recette_journaliere'].sum(), 2) if not df.empty else 0
    }
    
    reponse = gemini.chat_conseil(message, contexte)
    return jsonify({'reponse': reponse})


@app.route('/api/donnees-graphique')
def donnees_graphique():
    """Retourne les données pour les graphiques Chart.js."""
    df = get_dataframe_ventes()
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/articles')
def api_articles():
    """Liste tous les articles (pour le formulaire de saisie)."""
    articles = Article.query.all()
    return jsonify([a.to_dict() for a in articles])


# =====================================================================
# INITIALISATION
# =====================================================================

def creer_donnees_demo():
    """Crée quelques articles de démonstration si la base est vide."""
    if Article.query.count() == 0:
        articles_demo = [
            Article(nom='Pain', prix_unitaire=500, categorie='Boulangerie'),
            Article(nom='Lait (1L)', prix_unitaire=750, categorie='Laitier'),
            Article(nom='Riz (1kg)', prix_unitaire=600, categorie='Céréales'),
            Article(nom='Huile (1L)', prix_unitaire=1200, categorie='Épicerie'),
            Article(nom='Savon', prix_unitaire=300, categorie='Hygiène'),
        ]
        for a in articles_demo:
            db.session.add(a)
        db.session.commit()
        print("Articles de démonstration créés.")


if __name__ == '__main__':
    with app.app_context():
        db.create_all()          # Crée les tables si elles n'existent pas
        creer_donnees_demo()     # Données de démo
    
    app.run(debug=True, host='0.0.0.0', port=5000)
