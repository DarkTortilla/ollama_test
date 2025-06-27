X
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()


embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = HuggingFaceLLM(
    model_name="microsoft/DialoGPT-medium",
    tokenizer_name="microsoft/DialoGPT-medium",
    context_window=1024,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)

Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)

def search_documents(query: str) -> str:
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return str(response)


search_tool = FunctionTool.from_defaults(
    fn=search_documents,
    name="search_documents",
    description="Busca información específica en los documentos locales sobre React, desarrollo web o cualquier tema en los archivos."
)


agent = ReActAgent.from_tools(
    [search_tool],
    verbose=True,
    system_prompt="""Eres un asistente de desarrollo de software experto y amigable.

Tienes acceso a tres herramientas:
1. search_documents: Para buscar información específica en documentos locales

Instrucciones:
- Siempre intenta buscar primero en los documentos locales si la pregunta parece específica
- Si no encuentras información en los documentos, usa tu conocimiento general
- Si necesitas explicar un concepto técnico, usa la herramienta de explicación
- Combina información de múltiples fuentes cuando sea útil
- Sé claro, conciso y útil en tus respuestas
- Si no sabes algo, admítelo y sugiere alternativas

Responde en español de manera amigable y profesional."""
)

async def main():
    print(" Escribe 'salir' para terminar.")
    print("Puedes preguntar sobre:")
    print("- Contenido de tus documentos")
    print("-" * 50)
    
    while True:
        try:
            user_question = input("Tu pregunta: ").strip()
            
            if user_question.lower() in ['salir', 'exit', 'quit']:
                print(" Hasta luego")
                break
                
            if not user_question:
                print(" Por favor escribe una pregunta.")
                continue
            
            print(" Procesando...")
            
            # Procesar con el agente
            response = agent.chat(user_question)
            print(f"\n Respuesta: {response}")
            
        except KeyboardInterrupt:
            print("\n ¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Intenta con otra pregunta.")

if __name__ == "__main__":
    asyncio.run(main())