import os
import requests
from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Diretório temporário para armazenar as imagens geradas
TEMP_DIR = "/tmp/gradio"

@app.route('/run', methods=['GET'])
def run_model():
    # ... (código anterior para obter parâmetros)

    # Chamar a API Gradio
    client = Client("https://squaadai-sd-xl.hf.space/--replicas/yl24o/")
    result = client.predict(
        prompt, negative_prompt, prompt_2, negative_prompt_2,
        use_negative_prompt, use_prompt_2, use_negative_prompt_2,
        seed, width, height,
        guidance_scale_base, guidance_scale_refiner,
        num_inference_steps_base, num_inference_steps_refiner,
        apply_refiner,
        api_name="/run"
    )

    # Salvar a imagem em um diretório temporário
    image_path = os.path.join(TEMP_DIR, "image.png")
    result["image"].save(image_path)

    # Chamar sua API de salvar imagem para obter a URL s3
    save_image_url = "https://wosocial.bubbleapps.io/version-test/api/1.1/wf/save?file=image.png"
    files = {'file': open(image_path, 'rb')}
    response = requests.get(save_image_url, files=files)

    # Obter a URL s3 da resposta
    s3_url = response.json().get("url", "")

    # Remover o arquivo temporário após a conclusão
    os.remove(image_path)

    # Retornar a URL s3
    return jsonify({"s3_url": s3_url})

if __name__ == '__main__':
    # Criar o diretório temporário se não existir
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Executar o aplicativo Flask
    app.run(debug=True)
