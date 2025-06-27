import os
from dotenv import load_dotenv

print("=== VERIFICACIÓN DE VARIABLES DE ENTORNO ===")

# Cargar variables de entorno
load_dotenv()

# Verificar la API key
api_key = os.getenv("OPENAI_API_KEY")

print(f"1. Variable OPENAI_API_KEY encontrada: {'Sí' if api_key else 'No'}")

if api_key:
    print(f"2. Primeros 20 caracteres: {api_key[:20]}...")
    print(f"3. Últimos 4 caracteres: ...{api_key[-4:]}")
    print(f"4. Longitud total: {len(api_key)} caracteres")
    print(f"5. Comienza con 'sk-': {'Sí' if api_key.startswith('sk-') else 'No'}")
    
    # Verificar que no sea un valor de ejemplo
    if api_key == "tu-api-key-aqui":
        print("❌ ERROR: La API key aún tiene el valor de ejemplo 'tu-api-key-aqui'")
    elif api_key.startswith("sk-TU_API_KEY"):
        print("❌ ERROR: La API key aún tiene un valor de ejemplo")
    else:
        print("✅ La API key parece ser válida")
        
        # Intentar verificar con OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            print("6. Intentando verificar con OpenAI...")
            
            # Hacer una llamada simple para verificar
            response = client.models.list()
            print("✅ API key verificada exitosamente con OpenAI")
            
        except Exception as e:
            print(f"❌ Error al verificar con OpenAI: {e}")
else:
    print("❌ No se encontró la variable OPENAI_API_KEY")

print("\n=== VERIFICACIÓN DEL ARCHIVO .env ===")
env_file_path = ".env"
if os.path.exists(env_file_path):
    print(f"✅ Archivo .env encontrado en: {os.path.abspath(env_file_path)}")
    with open(env_file_path, 'r') as f:
        content = f.read()
        print("Contenido del archivo .env:")
        # Mostrar el contenido pero ocultar la API key real
        lines = content.split('\n')
        for line in lines:
            if line.startswith('OPENAI_API_KEY='):
                if 'sk-' in line:
                    print(f"OPENAI_API_KEY=sk-...{line.split('=')[1][-4:] if len(line.split('=')) > 1 else ''}")
                else:
                    print(line)
            else:
                print(line)
else:
    print(f"❌ Archivo .env NO encontrado en: {os.path.abspath(env_file_path)}")

print("\n=== DIRECTORIO DE TRABAJO ===")
print(f"Directorio actual: {os.getcwd()}")
print("Archivos en el directorio:")
for file in os.listdir('.'):
    print(f"  - {file}")
