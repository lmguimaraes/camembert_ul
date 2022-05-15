def _extraire_quoi(texte):
        reponseQuoi = ""
        for attribut in texte:
            if attribut['entity_group'] == 'ORG':
                reponseQuoi += attribut['word']+","
            if attribut['entity_group'] == 'MISC':
                reponseQuoi += attribut['word']+","
        if not reponseQuoi:
                return "Impossible déterminer"
        return reponseQuoi[:-1]  