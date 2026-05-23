import os
import pickle
import numpy as np
import faiss

from dotenv import load_dotenv

import vertexai
from vertexai.language_models import TextEmbeddingModel

# ------------------------------------
# Load Environment
# ------------------------------------
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# ------------------------------------
# Initialize Vertex AI
# ------------------------------------
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

# ------------------------------------
# Embedding Model
# ------------------------------------
embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-005"
)

# ------------------------------------
# Load FAISS Index
# ------------------------------------
index = faiss.read_index(
    "vectorstore/index.faiss"
)

# ------------------------------------
# Load Stored Chunks
# ------------------------------------
with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)

# ------------------------------------
# Retrieval Function
# ------------------------------------
def retrieve_context(query, top_k=3):

    query_embedding = embedding_model.get_embeddings(
        [query]
    )[0].values

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(chunks[idx])

    return "\n\n".join(retrieved_chunks)