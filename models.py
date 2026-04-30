from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Article(db.Model):
    """
    Table des articles disponibles à la vente.
    Chaque article a un nom, un prix unitaire et une catégorie.
    """
    __tablename__ = 'articles'
    
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100), nullable=False)
    prix_unitaire = db.Column(db.Float, nullable=False)
    categorie = db.Column(db.String(50), default='Général')
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relation : un article peut apparaître dans plusieurs lignes de commande
    lignes = db.relationship('LigneCommande', backref='article', lazy=True)
    
    def to_dict(self):
        """Convertit l'objet en dictionnaire pour l'API JSON."""
        return {
            'id': self.id,
            'nom': self.nom,
            'prix_unitaire': self.prix_unitaire,
            'categorie': self.categorie
        }


class Commande(db.Model):
    """
    Table des commandes journalières.
    Chaque commande correspond à une transaction client.
    """
    __tablename__ = 'commandes'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    client = db.Column(db.String(100), default='Client anonyme')
    total = db.Column(db.Float, default=0.0)
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relation : une commande contient plusieurs lignes
    lignes = db.relationship('LigneCommande', backref='commande', lazy=True, cascade='all, delete-orphan')
    
    def calculer_total(self):
        """Recalcule le total de la commande depuis ses lignes."""
        self.total = sum(ligne.sous_total for ligne in self.lignes)
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': str(self.date),
            'client': self.client,
            'total': self.total,
            'lignes': [l.to_dict() for l in self.lignes]
        }


class LigneCommande(db.Model):
    """
    Table des lignes de commande.
    Chaque ligne associe une commande à un article avec une quantité.
    C'est ici qu'on stocke : article + quantité + prix au moment de la vente.
    """
    __tablename__ = 'lignes_commande'
    
    id = db.Column(db.Integer, primary_key=True)
    commande_id = db.Column(db.Integer, db.ForeignKey('commandes.id'), nullable=False)
    article_id = db.Column(db.Integer, db.ForeignKey('articles.id'), nullable=False)
    quantite = db.Column(db.Integer, nullable=False, default=1)
    prix_unitaire = db.Column(db.Float, nullable=False)  # Prix au moment de la vente
    
    @property
    def sous_total(self):
        """Calcule le sous-total de cette ligne."""
        return self.quantite * self.prix_unitaire
    
    def to_dict(self):
        return {
            'id': self.id,
            'article': self.article.nom if self.article else 'Inconnu',
            'quantite': self.quantite,
            'prix_unitaire': self.prix_unitaire,
            'sous_total': self.sous_total
        }
