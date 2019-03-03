## server/app.py
from flask import Flask, jsonify, request
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import redis
import time

# declare constants
HOST = '0.0.0.0'
PORT = 8081

# initialize flask application
app = Flask(__name__)

# initialize redis store
cache = redis.Redis(host='redis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)


@app.route('/')
def hello():
    count = get_hit_count()
    parameters = request.get_json()
    return 'Hello World! I have been seen {} times.\n'.format(count)

# ML API
@app.route('/api/train', methods=['POST'])
def train():
    # get parameters from request
    parameters = request.get_json()
    
    # read iris data set
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # fit model
    C_param = parameters['C']
    clf = svm.SVC(C=float(C_param),
                  probability=True,
                  random_state=1)
    clf.fit(X, y)
    # persist model
    joblib.dump(clf, 'model.pkl')
    return jsonify({'accuracy': round(clf.score(X, y) * 100, 2), 'C': C_param })

if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)

