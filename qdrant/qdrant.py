import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, Qdrant
from qdrant_client import QdrantClient
from langchain.vectorstores import VectorStore
from dotenv import load_dotenv

load_dotenv()

os.chdir("qdrant/")

with open("north_america_cities_weather.txt", "r") as file:
    content = file.read()

chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

collection_name = "weather_data"

QdrantVectorStore.from_texts(
texts=chunks,
embedding=embeddings,
url=os.getenv("QDRANT_URL"),
api_key=os.getenv("QDRANT_API_KEY"),
collection_name=collection_name)
