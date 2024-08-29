import transformers
import torch

# Cargar el modelo
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Definir el contexto y la pregunta en español
messages = [
    {"role": "system", "content": "Eres un asistente útil que siempre responde de manera clara y precisa."},
    {"role": "user", "content": "¿Cuál es el clima actual en Boston?"},
]

# Ejecutar la generación de texto
outputs = pipeline(
    messages,
    max_new_tokens=100,  
    do_sample=True,
    top_p=0.95,  
    temperature=0.8,  
)
##
# Imprimir la respuesta generada
print(outputs[0]["generated_text"])
