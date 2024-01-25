from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

@app.route('/run', methods=['GET'])
def run_model():
    # Obter parâmetros da consulta da URL
    endpoint = request.args.get('endpoint', default='https://squaadai-sd-xl.hf.space/--replicas/yl24o/')
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

    # Chamar a API Gradio para o primeiro modelo
    client = Client(endpoint)
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

@app.route('/faceswapper', methods=['GET'])
def faceswapper():
    # Lógica para a segunda API (faceswapper)
    endpoint = request.args.get('endpoint', default='https://pierroromeu-faceswapper.hf.space/--replicas/eom4d/')
    user_photo = request.args.get('user_photo', default='')
    result_photo = request.args.get('result_photo', default='')
    name = request.args.get('name', default='')

    client = Client(endpoint)
    result = client.predict(
        user_photo, result_photo, name,
        api_name="/faceswapper"
    )

    return jsonify(result)

@app.route('/face_enhancer', methods=['GET'])
def face_enhancer():
    # Lógica para a terceira API (face_enhancer)
    endpoint = request.args.get('endpoint', default='https://pierroromeu-gfpgan.hf.space/')
    input_image = request.args.get('input_image', default='')
    version = request.args.get('version', default='v1.4')
    rescaling_factor = request.args.get('rescaling_factor', type=float, default=1)

    client = Client(endpoint)
    result = client.predict(
        input_image, version, rescaling_factor,
        api_name="/predict"
    )

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
