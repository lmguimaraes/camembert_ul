from camembert_model.camembert import CamembertModel
from acces_donnes.service_articles import ServiceArticles
from preprocessors.preprocessor_core_nlp import PreprocessorCoreNlp

camembert = CamembertModel()    
preprocessor = PreprocessorCoreNlp()
preprocessor.camembert = camembert

articles = ServiceArticles()
listeArticles = articles._obtenir_articles()

preprocessor._repondre_questions(listeArticles)
print('A')

