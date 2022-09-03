from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, make_response
from joblib import load as load_joblib

app = Flask(__name__)
pipeline = load_joblib("models/sentiments-v2.joblib")

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        frase = request.get_json()['message']
        vector = pipeline.named_steps.vectorizer.transform([frase])
        sentiment = pipeline.named_steps.model.predict(vector)[0]
        return make_response( sentiment, 200)

    except Exception as e:
        return make_response(str(e), 500)

@app.route('/webbrowser', methods=['POST'])
def webbrowser():
   frase = request.form.get('name')
   vector = pipeline.named_steps.vectorizer.transform([frase])
   sentiment = pipeline.named_steps.model.predict(vector)[0]

   if frase:
       print(f'{frase}: {sentiment}')
       return render_template('hello.html', name = sentiment)

   else:
       print('Request vazio -- redirecionando')
       return redirect(url_for('index'))




if __name__ == '__main__':
   app.run()