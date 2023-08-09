from flask import Flask, render_template, request
import sklearn
import pickle
import numpy as np

model = pickle.load(open('SVC.pkl', 'rb'))

app = Flask(__name__)

data = [-0.527497, 2.485389, -0.599278, -0.538926, -1.378495]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        mean_radius = float(request.form.get('radius'))
        mean_texture = float(request.form.get('texture'))
        mean_perimter = float(request.form.get('perimeter'))
        mean_area = float(request.form.get('area'))
        mean_smoothness = float(request.form.get('smoothness'))
        prediction = model.predict([[mean_radius, mean_texture, mean_perimter, mean_area, mean_smoothness]])

        if prediction[0]==1:
            pred="Cell is cancerous. Go and see your doctor"
        else:
            pred="Cell is not cancerous"
        return render_template('index.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
