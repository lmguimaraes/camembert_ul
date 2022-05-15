from transformers import pipeline

def _analyser_sentiment_texte(texte):
        negatif = positif = neutre = 0
        analyzer = pipeline(task='text-classification', model="cmarkea/distilcamembert-base-sentiment", tokenizer="cmarkea/distilcamembert-base-sentiment")
        results = analyzer(texte, return_all_scores=True)
        for result in results:
            for attr in result:
                if attr['label'] == '1 star' or attr['label'] == '2 stars':
                    negatif += attr['score']
                if attr['label'] == '3 stars':
                    neutre += attr['score']
                if attr['label'] == '4 stars' or attr['label'] == '5 stars':
                    positif += attr['score']
        if positif > negatif and positif > neutre:
            return "Sentiments positives"
        if negatif > positif and negatif > neutre:
            return "Sentiments negatives"
        else:
            return "Neutre"