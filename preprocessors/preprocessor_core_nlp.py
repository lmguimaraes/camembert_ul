from typing import List
from transformers import CamembertConfig
from camembert_model.camembert import CamembertModel


class PreprocessorCoreNlp:
    camembert = None  
    def _creer_donnees(self, textes):
        phrasesTokenized = []
        for texte in textes:
            phrases = self.camembert.tokenizer.tokenize(texte)
            if (len(phrases) > 1):
                phrases = [self.camembert.tokenizer.tokenize(phrase.replace('.', '')) for phrase in phrases]
                phrasesTokenized.append(phrases)
        return phrasesTokenized

