from transformers.modeling_camembert import CamembertForMaskedLM
from transformers.tokenization_camembert import CamembertTokenizer
from flaubert_model.TestCamembert import CamembertModel


camembert = CamembertModel()
camembert.Run()

tokenizer = transformers.CamembertTokenizer.from_pretrained("camembert-base")
model = transformers.CamembertForMaskedLM.from_pretrained("camembert-base")

model.eval()

masked_input = "Le camembert est <mask> :)"
print(camembert.RemplirMasque(masked_input, model, tokenizer, topk=3))
