import faiss
from sentence_transformers import SentenceTransformer
import pickle
from transformers import AutoTokenizer, T5ForConditionalGeneration


k=5

model = SentenceTransformer('all-MiniLM-L6-v2')
user_query= input("What do you want to ask? ")
query_embedding= model.encode(user_query)

index= faiss.read_index('Huberman Speech.index')
k = 4                         
D, I = index.search(query_embedding.reshape(1,-1), k) # sanity check
print(I)
print(D)

with open('text_segments.pkl','rb') as f:
    text_segments= pickle.load(f)              

#function for index to text

def index_to_text(index, text_segments):
    return [text_segments[i] for i in index[0]]

retrieved_segments= index_to_text(I, text_segments)

# Load pre-trained model
model_class= "google/flan-ul2"
tokenizer = AutoTokenizer.from_pretrained(model_class)
model = T5ForConditionalGeneration.from_pretrained(model_class)
context= " ".join(retrieved_segments)
input_text= f"Context: {context} Query: {user_query}"
# Preprocess the input text
inputs = tokenizer(input_text, return_tensors="pt")
# Generate the response
outputs = model.generate(inputs["input_ids"], max_length=400)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))