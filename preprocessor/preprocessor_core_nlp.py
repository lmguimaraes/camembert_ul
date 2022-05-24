import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import RegexpParser
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from camembert_model.camembert import CamembertModel

from analyse_sentiment import analyser_sentiment_texte
from reponses_questions.evaluer_candidats import evaluer_qui
from reponses_questions.evaluer_candidats import evaluer_ou
from reponses_questions.evaluer_candidats import evaluer_quoi
from preprocessor import arbre_syntaxique

class PreprocessorCoreNlp:
    def _repondre_questions(self, textes):
        chunker = RegexpParser("""
                       NP: {<DET>?<ADJ|ADJWH>*<NC|NPP>}       #To extract Noun Phrases
                       P: {<P>}                         #To extract Prepositions
                       V: {<V|VIMP|VINF.*>}             #To extract Verbs
                       PP: {<p> <NP>}          #To extract Prepositional Phrases
                       VP: {<V> <NP|PP>*}      #To extract Verb Phrases
                       """)
        syncTexte = 0
        textesTraitesNER = []
        tagsTextes = self._creer_donnees_tag(textes)
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
        model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
        nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        for texte in textes:
            re.sub(r'http\S+', '', texte)
            #print(analyser_sentiment_texte._analyser_sentiment_texte(texte))
            textesTraitesNER.append(nlp(texte))
        for texteTraite in textesTraitesNER:
            print("Qui: " + evaluer_qui._extraire_qui(texteTraite))
            print("Quoi: " + evaluer_quoi._extraire_quoi(texteTraite))
            print("Où: " + evaluer_ou._extraire_ou(texteTraite)+ "\n")
            arbre_syntaxique._separer_phrases_texte(tagsTextes[syncTexte])
            arbre = chunker.parse(tagsTextes[syncTexte])
            print("Après extraire\n", arbre)
            arbre.draw()
            print("---------------------------------------------------------------------------------------------------------------------------------------")
            syncTexte += 1

    def _creer_donnees_tag(self, textes):
        tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model")
        model = AutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")
        textesTraites = []
        tokenTagger = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)
        for texte in textes:
            listeTags = []  
            listeMots = []
            re.sub(r'http\S+', '', texte)
            tokenized = tokenTagger(texte)
            for token in tokenized:
                listeTags.append(token['entity_group'])
                listeMots.append(token['word'])
            textesTraites.append(list(zip(listeMots, listeTags)))
        return textesTraites 
        
    

    

        #quand = moment que l'événement central de l'article se produit
        #ajouter du traitement aux titres

    

    def _analyser_complexite_phrase(self, phrase):
        test = ''

        #séparer la partie 5w

        #5w corpus et lead
        #mettre en place une architecture d'apprentissage
        #rouler le crawler avec des differents cibles
        #mettre le doc des ref sur le cloud
        
        #parametres pour definir si fake news
        #IA vs journalistique

        #--mai
        #qualité des 5w
        #classification

