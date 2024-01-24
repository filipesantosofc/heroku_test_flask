from flask import Flask, request, jsonify
from gradio_client import Client
import requests
import os

app = Flask(__name__)

# Função para salvar a imagem no serviço de armazenamento
def save_image_to_api(image_path):
    api_url = "https://wosocial.bubbleapps.io/version-test/api/1.1/wf/save"
    files = {'file': open(image_path, 'rb')}
    response = requests.get(api_url, params={'file': image_path}, files=files)
    return response.json().get('url', '')

@app.route('/run', methods=['GET'])
def run_model():
    # ... (código anterior para obter parâmetros)

    # Chamar a API Gradio e salvar a imagem localmente
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

    # Salvar a imagem localmente
    image_path = "/tmp/gradio/output_image.png"
    result['image'].save(image_path)

    # Chamar a API para salvar a imagem e obter a URL
    saved_image_url = save_image_to_api(image_path)

    # Retornar a URL da imagem
    return jsonify({"saved_image_url": saved_image_url})

if __name__ == '__main__':
    app.run(debug=True)
