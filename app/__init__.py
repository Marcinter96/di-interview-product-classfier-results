from flask import Flask, request
import joblib
import json

def create_app():
    app = Flask(__name__)

    @app.route('/classify', methods=['POST'])
    def classify():
        data = request.get_json()
        title = data.get('title', '')

        classifier = joblib.load('./training/models/classifier_model_countVectorizer.joblib')
        
        count_vectorizer = joblib.load('./training/models/count_vectorizer.joblib')

        vectorized_text = count_vectorizer.transform([title])
    
        # Make predictions using the trained classifier
        prediction = classifier.predict_proba(vectorized_text)

        # Get the class labels from the classifier
        class_labels = classifier.classes_

        # Create a dictionary to store the class probabilities
        class_probabilities = {class_label: probability for class_label, probability in zip(class_labels, prediction[0])}

        # Sort the class probabilities in descending order
        sorted_probabilities = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)

        # Get the top 3 predictions
        top_3_predictions = sorted_probabilities[:3]

        # Convert the ndarray to a list
        product_type_predictions = classifier.predict(vectorized_text).tolist()

        # Prepare the response with the top 3 predictions
        response = {'title':title,
                    'top_3_results': [{'product_type': pred[0], 'score': round(pred[1],4)} for pred in top_3_predictions],
        'productType':product_type_predictions}

        # Serialize the response to JSON
        json_response = json.dumps(response, indent=4)
        return json_response

    return app