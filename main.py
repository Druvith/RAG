from Bert_encoder import CustomBertModel, Config
import torch
from transformers import BertTokenizer

# metal > cuda > (fall back) cpu
device = "mps" if torch.backends.mps.is_available else "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Initialise the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-large-cased')
model = CustomBertModel.from_pretrained(Config, device)

# inference mode
model.eval()

# generate the embeddings 
def generate_embeddings(text: str, model) -> torch.Tensor:
  tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
  tokenized_text = tokenized_text['input_ids']
  with torch.no_grad():
    embeddings = model(tokenized_text)
  embeddings = embeddings.squeeze(0)
  return embeddings, text


text = """Shree Krishna uses the word atha to indicate that Arjun may want to believe the other explanations that exist about the nature of the self. This verse needs to be understood in the context of the philosophical streams existing in India and their divergent understandings about the nature of self. Indian philosophy has historically comprised of twelve schools of thought. Six of these accept the authority of the Vedas, and hence they are called Āstik Darśhans. These are Mīmānsā, Vedānt, Nyāya, Vaiśheṣhik, Sānkhya, and Yog. Within each of these are more branches—for example, the Vedānt school of thought is further divided into six schools—Adavita vāda, Dwaita vāda, Viśhiṣhṭādvaita vāda, Viśhuddhadvaita vāda, Dwaitādvaita vāda, and Achintya-bhedābheda vāda. Each of these has further branches, for example, Advaita vāda is subdivided into Dṛiṣhṭi-sṛiṣhṭi vāda, Avachchheda vāda, Bimba-pratibimba vāda, Vivarta vāda, Ajāta vāda, etc. We will not go into the details of these schools here. Let it suffice for now to know that all these schools of thought accept the Vedas as the authority of reference. Accordingly, they all accept the eternal, unchangeable soul as the self."""

# test
embeddings, _ = generate_embeddings(text, model)
print(embeddings.shape)