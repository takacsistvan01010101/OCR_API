"""Filename: server.py
"""

import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')

        #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        test['Dependents'] = [str(x) for x in list(test['Dependents'])]

        #Getting the Loan_IDs separated out
        loan_ids = test['Loan_ID']

    except Exception as e:
        raise e

        from scipy.misc import imread, imresize
        import numpy as np
        x = imread('test1.png',mode='L')
        #compute a bit-wise inversion so black becomes white and vice versa
        x = np.invert(x)
        #make it the right size
        x = imresize(x,(28,28))
        #convert to a 4D tensor to feed into our model
        x = x.reshape(1,28,28,1)
        x = x.astype('float32')
        x /= 255

        #perform the prediction
        from keras.models import load_model
        model = load_model('cnn.h5')
        out = model.predict(x)

        final_predictions = np.argmax(out)

        """We can be as creative in sending the responses.
                But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)