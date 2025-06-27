from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
import os
from dotenv import load_dotenv
import torch

load_dotenv()

print("🚀 Iniciando chatbot inteligente...")

# Configurar embeddings (SIN LLM de OpenAI)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# NO configurar LLM global para evitar OpenAI
# Settings.llm = llm  # ← COMENTADO para evitar OpenAI
Settings.embed_model = embed_model

# Cargar documentos e indexar (solo embeddings, sin LLM)
try:
    print("📁 Cargando documentos...")
    documents = SimpleDirectoryReader('data').load_data()
    # Crear índice solo con embeddings (sin LLM)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    has_docs = True
    print("✅ Documentos indexados correctamente")
except Exception as e:
    print(f"⚠️ Sin documentos locales: {e}")
    has_docs = False
    index = None

class SmartChatbot:
    def __init__(self):
        print("✅ Chatbot configurado (solo embeddings + conocimiento base)")
        
        # Base de conocimiento expandida
        self.knowledge_base = {
            "react": {
                "definition": "React es una biblioteca de JavaScript desarrollada por Facebook para construir interfaces de usuario, especialmente para aplicaciones web de una sola página (SPA).",
                "features": ["Componentes reutilizables", "Virtual DOM", "JSX", "Hooks", "Unidirectional data flow"],
                "advantages": ["Rendimiento optimizado", "Ecosistema robusto", "Gran comunidad", "Fácil testing"],
                "concepts": ["State", "Props", "Components", "Lifecycle", "Context API"]
            },
            "javascript": {
                "definition": "JavaScript es un lenguaje de programación dinámico y versátil que se ejecuta tanto en navegadores como en servidores.",
                "features": ["Tipado dinámico", "Orientado a objetos", "Funcional", "Interpretado", "Event-driven"],
                "uses": ["Desarrollo web frontend", "Backend con Node.js", "Aplicaciones móviles", "Desktop apps"],
                "concepts": ["Variables", "Funciones", "Objetos", "Arrays", "Promises", "Async/Await"]
            },
            "python": {
                "definition": "Python es un lenguaje de programación de alto nivel, interpretado y de propósito general, conocido por su sintaxis clara y legible.",
                "features": ["Sintaxis simple", "Tipado dinámico", "Multiplataforma", "Orientado a objetos"],
                "uses": ["Desarrollo web", "Ciencia de datos", "Inteligencia artificial", "Automatización", "Scripting"],
                "frameworks": ["Django", "Flask", "FastAPI", "Pandas", "NumPy", "TensorFlow"]
            },
            "api": {
                "definition": "Una API (Application Programming Interface) es un conjunto de reglas y protocolos que permite que diferentes aplicaciones software se comuniquen entre sí.",
                "types": ["REST API", "GraphQL", "SOAP", "WebSocket"],
                "benefits": ["Reutilización de código", "Escalabilidad", "Separación de responsabilidades", "Integración"],
                "methods": ["GET", "POST", "PUT", "DELETE", "PATCH"]
            },
            "html": {
                "definition": "HTML (HyperText Markup Language) es el lenguaje de marcado estándar para crear páginas web.",
                "features": ["Elementos y etiquetas", "Estructura semántica", "Formularios", "Multimedia"],
                "versions": ["HTML5 es la versión actual", "Incluye APIs de JavaScript", "Soporte multimedia nativo"],
                "tags": ["div", "span", "header", "footer", "nav", "section", "article"]
            },
            "css": {
                "definition": "CSS (Cascading Style Sheets) es un lenguaje de hojas de estilo usado para describir la presentación de documentos HTML.",
                "features": ["Selectores", "Propiedades", "Flexbox", "Grid", "Animaciones", "Media queries"],
                "preprocessors": ["Sass", "Less", "Stylus"],
                "properties": ["color", "font-size", "margin", "padding", "display", "position"]
            },
            "node": {
                "definition": "Node.js es un entorno de ejecución de JavaScript construido sobre el motor V8 de Chrome que permite ejecutar JavaScript en el servidor.",
                "features": ["Event-driven", "Non-blocking I/O", "NPM package manager", "Cross-platform"],
                "uses": ["APIs REST", "Aplicaciones en tiempo real", "Microservicios", "Herramientas de build"],
                "modules": ["Express.js", "Socket.io", "Mongoose", "Lodash"]
            },
            "git": {
                "definition": "Git es un sistema de control de versiones distribuido que rastrea cambios en archivos y coordina el trabajo entre múltiples desarrolladores.",
                "commands": ["git clone", "git add", "git commit", "git push", "git pull", "git branch"],
                "concepts": ["Repository", "Branch", "Merge", "Pull request", "Conflict resolution"],
                "workflows": ["Feature branch", "Gitflow", "GitHub flow"]
            }
        }
    
    def search_documents_simple(self, query):
        """Búsqueda simple en documentos sin usar LLM"""
        if not has_docs:
            return None
        
        try:
            # Búsqueda por similitud semántica sin LLM
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(query)
            
            if nodes:
                results = []
                for node in nodes:
                    content = node.text.strip()
                    if len(content) > 50:  # Solo contenido sustancial
                        results.append(content)
                
                if results:
                    return "\n\n".join(results)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error buscando en documentos: {e}")
            return None
    
    def get_knowledge_response(self, question):
        """Busca respuesta en la base de conocimiento expandida"""
        question_lower = question.lower()
        
        for topic, info in self.knowledge_base.items():
            if topic in question_lower:
                response = f"**{topic.upper()}:**\n\n"
                response += f"📝 **Definición:** {info['definition']}\n\n"
                
                # Agregar todas las categorías disponibles
                for key, values in info.items():
                    if key != 'definition' and isinstance(values, list):
                        emoji_map = {
                            'features': '✨',
                            'uses': '🎯',
                            'advantages': '👍',
                            'types': '🔗',
                            'commands': '💻',
                            'concepts': '🧠',
                            'frameworks': '🛠️',
                            'methods': '🔧',
                            'tags': '🏷️',
                            'properties': '🎨',
                            'modules': '📦',
                            'workflows': '🔄'
                        }
                        
                        emoji = emoji_map.get(key, '•')
                        title = key.replace('_', ' ').title()
                        
                        response += f"{emoji} **{title}:**\n"
                        for item in values:
                            response += f"  • {item}\n"
                        response += "\n"
                
                return response
        
        return None
    
    def chat(self, question):
        """Función principal de chat - SIN OpenAI"""
        print("🤖 Procesando...")
        
        final_response = ""
        
        # 1. Buscar en documentos (sin LLM, solo similitud)
        if has_docs:
            doc_result = self.search_documents_simple(question)
            if doc_result:
                final_response += f"📚 **En tus documentos encontré:**\n{doc_result}\n\n"
        
        # 2. Buscar en base de conocimiento (siempre disponible)
        knowledge_result = self.get_knowledge_response(question)
        if knowledge_result:
            if final_response:
                final_response += f"🧠 **Información adicional:**\n{knowledge_result}"
            else:
                final_response += f"🧠 **Información general:**\n{knowledge_result}"
        
        # 3. Respuesta de emergencia
        if not final_response:
            final_response = self.get_fallback_response(question)
        
        return final_response
    
    def get_fallback_response(self, question):
        """Respuesta de emergencia"""
        available_topics = ", ".join(self.knowledge_base.keys())
        
        return f"""🤔 No encontré información específica sobre "{question}".

📚 **Temas disponibles:** {available_topics}

💡 **Ejemplos de preguntas:**
• "¿Qué es React?"
• "Explícame JavaScript"
• "¿Cómo funciona una API?"
• "¿Qué es Python?"
• "Comandos de Git"
• "Características de Node.js"

¿Podrías preguntar sobre alguno de estos temas?"""

async def main():
    chatbot = SmartChatbot()
    
    print("💬 ¡Chatbot listo! (Sin APIs de pago)")
    print("Combina: búsqueda en documentos + base de conocimiento")
    
    if has_docs:
        print("✅ Documentos disponibles para búsqueda semántica")
    else:
        print("⚠️ Sin documentos locales - usando solo conocimiento base")
    
    print("Puedes preguntar sobre: React, JavaScript, Python, APIs, HTML, CSS, Node.js, Git")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🤔 Tu pregunta: ").strip()
            
            if question.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
                
            if not question:
                print("⚠️ Por favor escribe una pregunta.")
                continue
            
            response = chatbot.chat(question)
            print(f"\n✨ **Respuesta:**\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())