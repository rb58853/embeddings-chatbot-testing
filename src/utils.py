from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    # Función para generar embeddings
    def generate_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings
    
    def similarity(self, text1, text2):
        # Embeddings de dos palabras o documentos
        embedding1 = np.array(self.generate_embeddings(text1))
        embedding2 = np.array(self.generate_embeddings(text2))

        # Redimensionar los arrays para que coincidan con la forma de entrada esperada de cosine_similarity
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)

        # Calcular la similitud del coseno
        return cosine_similarity(embedding1, embedding2)[0][0]

def similarity_by_vector(self, v1, v2):
    # Embeddings de dos palabras o documentos
    embedding1 = np.array(v1)
    embedding2 = np.array(v2)
    # Redimensionar los arrays para que coincidan con la forma de entrada esperada de cosine_similarity
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    # Calcular la similitud del coseno
    return cosine_similarity(embedding1, embedding2)[0][0] 