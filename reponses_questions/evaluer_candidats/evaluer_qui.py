def _extraire_qui(texte):
        reponseQui = ""
        for attribut in texte:
            if attribut['entity_group'] == 'PER':
                reponseQui += attribut['word']+","
        if not reponseQui:
                return "Impossible déterminer"
        return reponseQui[:-1]