

from langchain_core import embeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




load_dotenv()
embeddings = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=1536)

documents = ["sdfsdfsdf"
"sdfsdfsdfsdfsdfsdfsdf"]




query = "sdfsdfsdfsdfsdfsdfsdf"


doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)


scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index , score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[-1]
print(query,"is most similar to",documents[index])
print("Similarity score:",score)