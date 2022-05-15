def _separer_phrases_texte(texte):
        phrases = []
        phraseAjouter = []
        for token in texte:
                phraseAjouter.append(token)
                textetoken = token[0]
                if textetoken[-1] == '.' or textetoken[-1] == '?' or textetoken[-1] == '!' or textetoken[-1] == ';':
                    phrases.append(phraseAjouter)
                    phraseAjouter = []
        return phrases