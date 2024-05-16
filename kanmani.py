import pickle
from flask import Flask, request
from helper import clean_text
import json

# load the model
with open('Models/mnb_model.pkl', 'rb')as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict/', methods=['POST', 'GET'])
def predict() -> str:

    response = request.get_json()
    if isinstance(response, (list, tuple)):
        result = dict()
        for text in response:
            if len(clean_text(text)) > 1 :
                result[text] = model.predict([clean_text(text)])[0]
            else:
                result[text] = 'Not an Arabic word'
        return json.dumps(result)
    elif response == None:
        return json.dumps("Make sure you're not sending None and sending the request with the Content-type: application/json header.")

    return json.dumps("Make sure you're sending a sequence(list/tuple) of strings and sending ")


app.run(debug=True, port=8080) #run app on port 8080 in debug mode
