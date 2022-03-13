import re
import this
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize, RegexpParser
from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from camembert_model.camembert import CamembertModel

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
            textesTraitesNER.append(nlp(texte))
        for texteTraite in textesTraitesNER:
            print("Qui: " + self._extraire_qui(texteTraite))
            print("Quoi: " + self._extraire_quoi(texteTraite))
            print("Où: " + self._extraire_ou(texteTraite)+ "\n")
            arbre = chunker.parse(tagsTextes[syncTexte])
            #print("Après extraire\n", arbre)
            #arbre.draw()
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

    def _extraire_qui(self, texte):
        reponseQui = ""
        for attribut in texte:
            if attribut['entity_group'] == 'PER':
                reponseQui += attribut['word']+","
        if not reponseQui:
                return "Impossible déterminer"
        return reponseQui[:-1]    
        
    def _extraire_ou(self, texte):
        reponseOu = ""
        for attribut in texte:
            if attribut['entity_group'] == 'LOC':
                reponseOu += attribut['word']+","
        if not reponseOu:
                return "Impossible déterminer"
        return reponseOu[:-1]   

    def _extraire_quoi(self, texte):
        reponseQuoi = ""
        for attribut in texte:
            if attribut['entity_group'] == 'ORG':
                reponseQuoi += attribut['word']+","
            if attribut['entity_group'] == 'MISC':
                reponseQuoi += attribut['word']+","
        if not reponseQuoi:
                return "Impossible déterminer"
        return reponseQuoi[:-1]  

