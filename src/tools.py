from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import json

load_dotenv()

tavily_search = TavilySearchResults(max_results=3, api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_weather_south_america(capital: str) -> str:
    """
    Returns the weather status, including temperature and humidity, for a given South American capital.

    Args:
        capital (str): The name of the capital city.

    Returns:
        str: A formatted string describing the temperature and humidity in the specified capital.
             If the capital is not recognized, returns an error message.

    Example:
        >>> weather_south_america("Buenos Aires")
        'The temperature in Buenos Aires is 35 ºC and the relative humidity is 50 %'

    Supported capitals:
        - Buenos Aires
        - Santiago de Chile
        - Lima
    """
    weather_data = {
        "Buenos Aires": "The temperature in Buenos Aires is 35 ºC and the relative humidity is 50 %",
        "Santiago de Chile": "The temperature in Santiago de Chile is 40 ºC and the relative humidity is 60 %",
        "Lima": "The temperature in Lima is 30 ºC and the relative humidity is 50 %"
    }

    return weather_data.get(capital, "Weather data for this capital is not available.")

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

collection_name = "weather_data"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

vector_store = QdrantVectorStore(
    client=qdrant_client, 
    collection_name=collection_name, 
    embedding=embeddings
)

@tool
def get_weather_north_america(user_query: str) -> dict:
    """
    Searches for the weather status, including temperature and humidity, for specific North American capitals 
    in the Qdrant vector store based on the user's query.

    This function retrieves the weather information for the following capitals:
    - Boston (Massachusetts)
    - Phoenix (Arizona)
    - Sacramento (California)

    Args:
        user_query (str): The user's query related to weather information for a specific capital city.

    Returns:
        dict: A dictionary containing the retrieved weather information and metadata for the specified capital.
              If the capital is not recognized, returns a message indicating that the weather data is not available.

    Example:
        >>> get_weather_north_america("What is the weather in Boston?")
        {
            "documents": [
                {
                    "content": "The temperature in Boston (Massachusetts) is 35 ºC and the relative humidity is 50 %.",
                    "metadata": {"source": "weather_api"}
                }
            ]
        }
    """
    search_results = vector_store.similarity_search(query=user_query)

    # Convert the results to a JSON-serializable format
    serializable_results = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in search_results
    ]

    # Return the results as a dictionary
    return {"documents": serializable_results}





