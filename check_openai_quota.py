import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    models = client.models.list()
    print("API key válida - Conexión exitosa")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("Quota disponible - Tu cuenta tiene créditos")
    print(f"Respuesta de prueba: {response.choices[0].message.content}")
    
except Exception as e:
    if "insufficient_quota" in str(e):
        print("Sin créditos - Necesitas agregar fondos a tu cuenta OpenAI")
        print("Ve a: https://platform.openai.com/account/billing")
    else:
        print(f"Error: {e}")
