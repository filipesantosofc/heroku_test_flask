from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados da solicitação POST
        data = request.json

        # Extrair os parâmetros necessários
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        # Adicione mais parâmetros conforme necessário

        # Fazer a previsão usando o Gradio
        client = Client("https://squaadai-sd-xl.hf.space/--replicas/yl24o/run/predict")
        result = client.predict(prompt, negative_prompt)

        # Retornar o resultado como JSON
        return jsonify({"result": result})

    except Exception as e:
        # Em caso de erro, retornar uma mensagem de erro
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Definir o host como 0.0.0.0 para que seja acessível externamente
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
