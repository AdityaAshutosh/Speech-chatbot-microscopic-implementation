from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import pickle

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embedding(text):
    return model.encode(text)

json_file_path= './json/processed_data.json'
with open(json_file_path, 'r') as f:
    processed_data = json.load(f)

text_segments= [item['processed_text'] for item in processed_data]

# Create embeddings for each processed text item from json
embeddings = [create_embedding(item['processed_text']) for item in processed_data]

# Convert to numpy array
embeddings_array = np.array(embeddings).astype('float32')

# Create the index
dimension = embeddings_array.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)

# Feed the numpy array to the database
index.add(embeddings_array)

# Save the index
faiss.write_index(index, "Huberman Speech.index")

print("Embeddings created and added to Faiss index.")

with open('text_segments.pkl', 'wb') as f:
    pickle.dump(text_segments, f)

with open('text_segments.pkl','rb') as f:
    text_segments= pickle.load(f)

def map_index_to_text(idx):
    return text_segments[idx]

