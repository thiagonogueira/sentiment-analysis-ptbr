from joblib import load as load_joblib


if __name__ == '__main__':
    
    pipeline = load_joblib("models/sentiments-v2.joblib")
    
    while True:
        
        msg = input("Digite sua frase: ")
        
        vector = pipeline.named_steps.vectorizer.transform([msg])
        res = pipeline.named_steps.model.predict(vector)
        
        print("Predição: {}".format(res[0]))
