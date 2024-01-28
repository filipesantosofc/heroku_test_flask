from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

@app.route('/run', methods=['GET'])
def run_model():
    # Obter parâmetros da consulta da URL
    endpoint = request.args.get('endpoint', default='https://squaadai-sd-xl.hf.space/--replicas/yl24o/')
    hf_token = request.args.get('hf_token', default='')

    # Restante dos parâmetros específicos para a primeira API
    prompt = request.args.get('prompt', default='')
    negative_prompt = request.args.get('negative_prompt', default='')
    prompt_2 = request.args.get('prompt_2', default='')
    negative_prompt_2 = request.args.get('negative_prompt_2', default='')
    use_negative_prompt = request.args.get('use_negative_prompt', type=bool, default=True)
    use_prompt_2 = request.args.get('use_prompt_2', type=bool, default=True)
    use_negative_prompt_2 = request.args.get('use_negative_prompt_2', type=bool, default=True)
    seed = request.args.get('seed', type=int, default=0)
    width = request.args.get('width', type=int, default=256)
    height = request.args.get('height', type=int, default=256)
    guidance_scale_base = request.args.get('guidance_scale_base', type=float, default=1.0)
    guidance_scale_refiner = request.args.get('guidance_scale_refiner', type=float, default=1.0)
    num_inference_steps_base = request.args.get('num_inference_steps_base', type=int, default=10)
    num_inference_steps_refiner = request.args.get('num_inference_steps_refiner', type=int, default=10)
    apply_refiner = request.args.get('apply_refiner', type=bool, default=True)

    # Chamar a API Gradio
    client = Client(endpoint, hf_token=hf_token)
    result = client.predict(
        prompt, negative_prompt, prompt_2, negative_prompt_2,
        use_negative_prompt, use_prompt_2, use_negative_prompt_2,
        seed, width, height,
        guidance_scale_base, guidance_scale_refiner,
        num_inference_steps_base, num_inference_steps_refiner,
        apply_refiner,
        api_name="/run"
    )

    return jsonify(result)

@app.route('/predict', methods=['GET'])
def predict_gan():
    # Obter parâmetros da consulta da URL
    endpoint = request.args.get('endpoint', default='https://pierroromeu-gfpgan.hf.space/--replicas/dgwcd/')
    hf_token = request.args.get('hf_token', default='')
    filepath = request.args.get('filepath', default='')
    version = request.args.get('version', default='v1.4')
    rescaling_factor = request.args.get('rescaling_factor', type=float, default=2.0)

    # Chamar a API Gradio
    client = Client(endpoint, hf_token=hf_token)
    result = client.predict(
        filepath,
        version,
        rescaling_factor,
        api_name="/predict"
    )

    return jsonify(result)

@app.route('/faceswapper', methods=['GET'])
def faceswapper():
    # Obter parâmetros da consulta da URL
    endpoint = request.args.get('endpoint', default='https://pierroromeu-faceswapper.hf.space/--replicas/u42x7/')
    hf_token = request.args.get('hf_token', default='')
    user_photo = request.args.get('user_photo', default='')
    result_photo = request.args.get('result_photo', default='')
    name_for_saving = request.args.get('name_for_saving', default='')

    # Chamar a API Gradio
    client = Client(endpoint, hf_token=hf_token)
    result = client.predict(
        user_photo,
        result_photo,
        name_for_saving,
        api_name="/faceswapper"
    )

    return jsonify(result)

# run.py
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=9000)
