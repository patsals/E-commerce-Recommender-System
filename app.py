from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np 
import json 

app = Flask(__name__)
model = tf.keras.models.load_model('saved_model')

user_mapper_file_name = 'user_id_mapper.json'
with open(user_mapper_file_name, 'r') as json_file:
    uid_mapper = json.load(json_file)

product_mapper_file_name = 'product_id_mapper.json'
with open(product_mapper_file_name, 'r') as json_file:
    pid_mapper = json.load(json_file)

user_product_interactions_file_name = 'user_product_interactions.json'
with open(user_product_interactions_file_name, 'r') as json_file:
    user_product_interactions = json.load(json_file)
    user_product_interactions = {eval(key):value for key,value in user_product_interactions.items()}

@app.route('/')
def main():
    users = list(uid_mapper.keys())
    random_users = np.random.choice(users, size=10, replace=False)
    return render_template('login.html', options=random_users)


@app.route('/home', methods=['POST'])
def home():
    uid_input = request.form['uid']

    uid_input_mapped = uid_mapper[uid_input]
    unseen_products = [product for product in pid_mapper.values() if (uid_input, product) not in user_product_interactions]

    num_products = len(pid_mapper)
    users = np.array([int(uid_input_mapped)] * num_products, dtype=np.int64)
    products = np.array(unseen_products, dtype=np.int64)
    predictions = model.predict([users, products])

    products_predictions = list(zip(pid_mapper.keys(), map(lambda x: x[0], predictions)))
    products_predictions.sort(key=lambda x: x[1], reverse=True)
    
    return render_template('home.html', data=[uid_input,products_predictions[:5]])


if __name__ == "__main__":
    app.run(debug=True)
