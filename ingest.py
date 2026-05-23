import os
import pickle
import numpy as np
import faiss

from dotenv import load_dotenv
from PyPDF2 import PdfReader

import vertexai
from vertexai.language_models import TextEmbeddingModel

# ------------------------------------
# Load Environment Variables
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
# Load Embedding Model
# ------------------------------------
embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-005"
)

DATA_FOLDER = "data"

all_chunks = []

# ------------------------------------
# Read PDFs
# ------------------------------------
for file in os.listdir(DATA_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DATA_FOLDER, file)

        print(f"Reading: {file}")

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        # --------------------------------
        # Chunking
        # --------------------------------
        chunk_size = 800

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            all_chunks.append(chunk)

print(f"\nTotal Chunks: {len(all_chunks)}")

# ------------------------------------
# Create Embeddings
# ------------------------------------
embeddings = []

for chunk in all_chunks:

    response = embedding_model.get_embeddings([chunk])

    vector = response[0].values

    embeddings.append(vector)

print("Embeddings created")

# ------------------------------------
# Convert to NumPy
# ------------------------------------
embeddings = np.array(
    embeddings,
    dtype="float32"
)

# ------------------------------------
# Create FAISS Index
# ------------------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# ------------------------------------
# Save FAISS Index
# ------------------------------------
faiss.write_index(
    index,
    "vectorstore/index.faiss"
)

# ------------------------------------
# Save Chunks
# ------------------------------------
with open(
    "vectorstore/chunks.pkl",
    "wb"
) as f:

    pickle.dump(all_chunks, f)

print("\nVector Database Saved Successfully")