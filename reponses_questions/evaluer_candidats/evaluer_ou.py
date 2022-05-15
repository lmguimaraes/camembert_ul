def _extraire_ou(texte):
        reponseOu = ""
        for attribut in texte:
            if attribut['entity_group'] == 'LOC':
                reponseOu += attribut['word']+","
        if not reponseOu:
                return "Impossible déterminer"
        return reponseOu[:-1]   