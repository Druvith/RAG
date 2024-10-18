from Bert_encoder import CustomBertModel, Config
import torch
from transformers import BertTokenizer
import random
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

torch.manual_seed(1337)
# test: 1
# Define test cases
test_cases = [
    ("You're dog's cute!", "Lovely Dog!", "Similar, but different wording"),
    ("You're dog's cute!", "You're dog's cute!", "Identical sentences"),
    ("You're dog's cute!", "The sky is blue today.", "Completely different context"),
    ("He likes cats.", "She loves cats.", "Different pronouns, similar meaning"),
    ("The book was amazing!", "I didn't like the movie.", "Opposite sentiments"),
]

# Run tests for custom implementation and print similarity scores
print("Similarity Scores:")
for text1, text2, description in test_cases:
    embedding1, _ = generate_embeddings(text1.lower(), model)
    embedding2, _ = generate_embeddings(text2.lower(), model)
    similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=1)
    similarity = similarity.to("cpu")
    print(f"Similarity between '{text1}' and '{text2}' ({description}): {similarity.item():.3f}")