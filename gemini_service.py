# gemini_service.py
# Ce fichier gère toute la communication avec l'API Gemini de Google.
# Gemini analyse les données et génère des conseils personnalisés.

import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Configuration de l'API Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


class ServiceGemini:
    """
    Interface avec Gemini 1.5 Flash (modèle rapide et gratuit).
    Génère des conseils de gestion basés sur les données réelles.
    """
    
    def __init__(self):
        # On utilise gemini-1.5-flash pour la rapidité et les quotas gratuits
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyser_ventes(self, donnees_resume: dict) -> dict:
        """
        Envoie un résumé des ventes à Gemini et reçoit des conseils.
        
        Paramètre donnees_resume : dictionnaire avec les statistiques clés
        Retourne : conseils structurés en JSON
        """
        
        # Construction du prompt (instruction pour Gemini)
        prompt = f"""
        Tu es un expert en gestion commerciale et analyse de ventes pour une petite entreprise.
        
        Voici les données de ventes récentes :
        {json.dumps(donnees_resume, ensure_ascii=False, indent=2)}
        
        Analyse ces données et fournis une réponse JSON avec EXACTEMENT cette structure :
        {{
            "diagnostic": "Résumé en 2 phrases de la situation actuelle",
            "tendance": "hausse" ou "baisse" ou "stable",
            "risques": ["risque 1", "risque 2", "risque 3"],
            "opportunites": ["opportunité 1", "opportunité 2"],
            "conseils_immediats": ["conseil urgent 1", "conseil urgent 2", "conseil urgent 3"],
            "conseils_moyen_terme": ["conseil 1 pour les 2 prochaines semaines", "conseil 2"],
            "articles_a_privilegier": ["article ou catégorie recommandée"],
            "alerte_penurie": true ou false,
            "score_sante": un nombre entre 0 et 100,
            "prediction_semaine": "Prévision pour la semaine prochaine en 1 phrase"
        }}
        
        Réponds UNIQUEMENT avec le JSON, sans texte supplémentaire.
        """
        
        try:
            reponse = self.model.generate_content(prompt)
            texte = reponse.text.strip()
            
            # Nettoyage du JSON (Gemini peut ajouter des balises markdown)
            if texte.startswith('```'):
                texte = texte.split('```')[1]
                if texte.startswith('json'):
                    texte = texte[4:]
            
            return json.loads(texte)
        
        except json.JSONDecodeError:
            # Si Gemini ne retourne pas un JSON valide, on retourne une réponse par défaut
            return {
                'diagnostic': 'Analyse en cours. Ajoutez plus de données pour de meilleurs conseils.',
                'tendance': 'stable',
                'risques': ['Données insuffisantes pour une analyse complète'],
                'opportunites': ['Continuez à saisir vos ventes quotidiennement'],
                'conseils_immediats': ['Saisir les ventes chaque jour', 'Vérifier les stocks', 'Observer les articles populaires'],
                'conseils_moyen_terme': ['Analyser les tendances hebdomadaires'],
                'articles_a_privilegier': ['À déterminer avec plus de données'],
                'alerte_penurie': False,
                'score_sante': 50,
                'prediction_semaine': 'Prévision disponible après 7 jours de données'
            }
        
        except Exception as e:
            return {'erreur': f'Erreur API Gemini : {str(e)}', 'score_sante': 0}
    
    def chat_conseil(self, message: str, contexte: dict) -> str:
        """
        Chat interactif : l'utilisateur peut poser des questions à Gemini
        sur son activité commerciale.
        """
        prompt = f"""
        Tu es un assistant commercial expert. Contexte de l'entreprise :
        {json.dumps(contexte, ensure_ascii=False)}
        
        Question de l'utilisateur : {message}
        
        Réponds de façon concise, pratique et en français.
        Donne des conseils actionnables adaptés aux données ci-dessus.
        """
        
        try:
            reponse = self.model.generate_content(prompt)
            return reponse.text
        except Exception as e:
            return f"Erreur lors de la communication avec Gemini : {str(e)}"
