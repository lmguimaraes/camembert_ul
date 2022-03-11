import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, RegexpParser
from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from camembert_model.camembert import CamembertModel




class PreprocessorCoreNlp:
    def _creer_donnees(self, textes):
        tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model")
        model = AutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")
        texteTraite = []
        tokenTagger = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)
        for texte in textes:
            re.sub(r'http\S+', '', texte)
            texteTraite.append(tokenTagger(texte))   
        return texteTraite

    def _repondre_questions(self, textes):
        texteTraite = []
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
        model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
        nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        for texte in textes:
            re.sub(r'http\S+', '', texte)
            texteTraite.append(nlp(texte))
        return texteTraite

