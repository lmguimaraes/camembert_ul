import torch

class CamembertModel:
    def Run(self):
        camembert = torch.hub.load('pytorch/fairseq', 'camembert')
        self.ExtraireCaracterisques(camembert)          
            
    def Masque(self, camembert, masked_line, topk=3):
        #MLM (Masked Language Modeling) utilisé pour prévoir les mots masqués. Utiliser avec <mask>

        #Exemple:
        #masked_line = 'Le camembert est <mask> :)'
        #camembert.fill_mask(masked_line, topk=3)
           
        camembert.fill_mask(masked_line, topk)  

    def Evaluation(self, camembert):
        #Utilisé pour évaluer et peaufiner l'entraînement
        camembert.eval()

    def ExtraireCaracterisques(self, camembert):
        #Extraire les caractéristiques de la dernière couche
        line = "J'aime le camembert !"
        tokens = camembert.encode(line)
        last_layer_features = camembert.extract_features(tokens)
        assert last_layer_features.size() == torch.Size([1, 10, 768])

        # Extraire les caractéristiques de toutes les couches
        all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
        assert len(all_layers) == 13

    def RemplirMasque(masked_input, model, tokenizer, topk=5):
        # Adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
        assert masked_input.count("<mask>") == 1
        input_ids = torch.tensor(tokenizer.encode(masked_input, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        logits = model(input_ids)[0]  # The last hidden-state is the first element of the output tuple
        masked_index = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
        logits = logits[0, masked_index, :]
        prob = logits.softmax(dim=0)
        values, indices = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = " ".join(
            [tokenizer.convert_ids_to_tokens(indices[i].item()) for i in range(len(indices))]
        )
        masked_token = tokenizer.mask_token
        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(" ")):
            predicted_token = predicted_token_bpe.replace("\u2581", " ")
            if " {0}".format(masked_token) in masked_input:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(" {0}".format(masked_token), predicted_token),
                        values[index].item(),
                        predicted_token,
                    )
                )
            else:
                topk_filled_outputs.append(
                    (masked_input.replace(masked_token, predicted_token), values[index].item(), predicted_token,)
                )
        return topk_filled_outputs