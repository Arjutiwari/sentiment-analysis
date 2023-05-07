from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder=r'C:\Users\apurb\Desktop\combined\templates')

categorical_to_string = {
    0: 'Negative',
    1: 'Positive'
}

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        model = pickle.load(open('model.pkl', 'rb'))
        vectorizer = pickle.load(open("vector.pkl", "rb"))

        transformed_reviews = vectorizer.transform(df['Reviews'])
        predictions = model.predict(transformed_reviews)
        df['Sentiment'] = predictions

        results_dict = {}
        for product, group in df.groupby('Product'):
            num_reviews = len(group)
            num_positives = np.count_nonzero(group['Sentiment'])
            num_negatives = num_reviews - num_positives
            percent_positive = num_positives / num_reviews * 100
            percent_negative = 100 - percent_positive

            results_dict[product] = {
                'num_reviews': num_reviews,
                'num_positives': num_positives,
                'num_negatives': num_negatives,
                'percent_positive': percent_positive,
                'percent_negative': percent_negative
            }

        return render_template('index2.html', results_dict=results_dict)

    return render_template('index1.html')



@app.route('/model3')
def model1():
    return render_template('model3.html')

@app.route('/model4')
def model2():
    return render_template('model4.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
