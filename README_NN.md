# Deploying a Neural Network Classifier

### Graduate Program in Electrical and Computer Engineering

#### Department of Computer Engineering and Automation 

##### EEC1509 Machine Learning

This repo is a test suite for the skills acquired in the [Fundamentals of Deep Learning](https://github.com/ivanovitchm/ppgeecmachinelearning) lessons to deploy a classification model based on neural networks on the publicly available [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

The results here presented are a continuation of a previous work [Using MNSIT data to train and deploy a classification model ](https://github.com/). The main goal of this project is to deploy a neural network model using the [FastAPI](https://fastapi.tiangolo.com/) module, creating an API and tests. The API tests are going to be incorporated into a CI/CD framework using GitHub Actions. The live API will be deployed using [Heroku](https://www.heroku.com/). [Weights & Biases](https://wandb.ai/) will be used to manage, track and optimize the models.

<center><img width="800" src="images/deploy.png"></center>

## Environment Setup

All the necessary modules can be found in the ``requirements.txt`` file, and can be installed by running:

```bash
pip install -r requirements.txt
```

## Dataset

The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 

<center><img width="800" src="images/mnist_sample.jpg"></center>

## Model development

The hardest parts have been done for us thanks to the Keras library and the National Institute of Standards and Technology (the NIST of MNIST). The information has been gathered and is ready for processing. 

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

The model developed here does not accept the $28 times 28$ matrices provided by the Keras module, so data flattening is required. This process is not ideal because it can withhold information about neighboring pixels, but it is adequate for building a baseline model. 

The retrieved data will then be flattened to produce several vectors of length 784 $(28 \cdot 28)$: 

```python
def reshape(array: np.array) -> np.array:
    """
        The samples in the input array are faltered. 
    """
    samples, w, h = array.shape

    return array.reshape((samples, w * h))

x_train = reshape(x_train)
x_test = reshape(x_test)
```

The sample distributions in the explored sets are depicted below: 

<center><img width="800" src="images/dataset_distribution.png"></center>

A classification model, in this case a Multi Layer Perceptron (MLP), was constructed using the [Tensorflow](https://www.tensorflow.org/) module by:

```python
mlp = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.15),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(10, activation='softmax')
])

mlp.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
```

A graphical representation of the proposed MLP model is shown in the figure below. 

<center><img width="800" src="images/nn.jpg"></center>

The model was then fitted to the data using two callbacks: the `EarlyStopping`, which stops the training process when the model's validation loss does not improve, and the `ReduceLROnPlateau`, which monitors the validation loss and reduces the learning rate when it reaches a plateau. 

```python
history = mlp.fit(
    x_train, y_train,
    validation_split=0.1,
    batch_size=64,
    epochs=2000,
    shuffle=True,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)
    ]
)
```

The figure below depicts the proposed classification model's loss function and accuracy values during the training phase. The loss function was successfully minimized with values approaching zero for the training set and low values for the validation set, indicating success in this step. The accuracy curve depicts a similar scenario, with both curves exceeding 90%. 

<center><img width="800" src="images/mlp_training.png"></center>

In order to follow the performance of machine learning experiments, the project marked certain stage outputs of the data pipeline as metrics. The metrics adopted here are: accuracy, f1, precision, recall. The results for the trained model are shown bellow.

|     Class    | Precision | Recall | F1-Score | Samples |
|:------------:|:---------:|:------:|:--------:|:-------:|
|       0      |    0.99   |  0.99  |   0.99   |   980   |
|       1      |    0.99   |  0.99  |   0.99   |   1135  |
|       2      |    0.98   |  0.98  |   0.98   |   1032  |
|       3      |    0.98   |  0.99  |   0.98   |   1010  |
|       4      |    0.99   |  0.98  |   0.98   |   982   |
|       5      |    0.99   |  0.98  |   0.98   |   892   |
|       6      |    0.99   |  0.99  |   0.99   |   958   |
|       7      |    0.98   |  0.99  |   0.98   |   1028  |
|       8      |    0.98   |  0.98  |   0.98   |   974   |
|       9      |    0.98   |  0.98  |   0.98   |   1009  |
|              |           |        |          |         |
|   Accuracy   |           |        |   0.98   |  10000  |
|   Macro avg  |    0.98   |  0.98  |   0.98   |  10000  |
| Weighted avg |    0.98   |  0.98  |   0.98   |  10000  |

The confusion matrix of the model's classifications is shown below. 

<center><img width="800" src="images/mlp_cf_matrix.png"></center>

The confusion matrix demonstrates the success of the classification approach used in this project, with only a few samples mislabeled. This model can be improved in the future and the inference model saved to the Weights & Biases platform for live replacement in the final application. 

## Saving the model to Weights & Biases

Using the following code, this training run can be saved in the Weights & Biases platform for experiment tracking: 

```python
run = wandb.init(project='proj_mnist', job_type='train')

print('Evaluating Model...')

fbeta = fbeta_score(y_test, y_pred, beta=1, zero_division=1, average='weighted')
precision = precision_score(y_test, y_pred, zero_division=1, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=1, average='weighted')
acc = accuracy_score(y_test, y_pred)

print(f'Accuracy: {acc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {fbeta}')

run.summary['Acc'] = acc
run.summary['Precision'] = precision
run.summary['Recall'] = recall
run.summary['F1'] = fbeta

print('Uploading confusion matrix...')

run.log({
    'confusion_matrix': wandb.Image(fig)
})


# ROC curve
predict_proba = mlp.predict(x_test)
wandb.sklearn.plot_roc(y_test, predict_proba, np.unique(y_train))

run.finish()
```


## Introduction to FastAPI

**FastAPI** is a modern API framework that relies heavily on type hints for its capabilities.

As the name suggests, FastAPI is designed to be fast in execution and also in development. It is built for maximum flexibility in that it is solely an API. You are not tied into particular backends, frontends, etc. Thus enabling composability with your favorite packages and/or existing infrastructure.

Getting started is as simple as writing a main.py containing:

```python
from fastapi import FastAPI

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get('/')
async def say_hello():
    return {'greeting': 'Hello World!'}
```

To run the app, [uvicorn](https://www.uvicorn.org/) can be used in the shell: ```uvicorn source.main:app --reload```. 

> Uvicorn is an ASGI (Asynchronous Server Gateway Interface) web server implementation for Python. 

By default, our app will be available locally at ```http://127.0.0.1:8000```. The ```--reload``` allows you to make changes to your code and have them instantly deployed without restarting *uvicorn*. For further reading the [FastAPI docs](https://fastapi.tiangolo.com/) are excellently written, check them out!

When a user makes a get request in the root route for this project, a static page will be displayed.
The FastAPI already supports this use case, and the sample code can be modified to meet these requirements:  

```python
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

# Instantiate the app.
app = FastAPI()

# Servers a static page for requests to the root endpoint.
app.mount('/', StaticFiles(directory='./source/static', html = True), name='static')
```

The application's home page is displayed below. 

<center><img width="800" src="images/index2.png"></center>

## Including an inference route in the API 

Using the decorator ``app.post``, a new POST route can be incorporated to the API, which will receive the image capture of the number drawn by the user in the canvas and return the classification probabilities for each class. 

The Weights & Biases platform will be used to retrieve the classification model.
This can be a specific version of the most recent one, ensuring that any changes to the model are continuously integrated into the application. 

```python
# Define a GET on the specified endpoint.
@app.post('/predict_nn/')
async def predict_nn(drawing_data: str) -> dict:
    """
        Receive the base64 encoding of an image 
        containing a handwritten digit and make 
        predictions using an ml model.
        
        Args:
            drawing_data (str): Base64 encoding of an image.

        Returns:
            dict: JSON response containing the neural network predictions
    """
    # Convert data in url to numpy array
    img_str = re.search(r'base64,(.*)', drawing_data.replace(' ', '+')).group(1)
    img_bytes = io.BytesIO(base64.b64decode(img_str))
    img = Image.open(img_bytes)

    # Normalize pixel values
    input = np.array(img)[:, :, 0:1].reshape((1, 28, 28)) / 255.0

    model = load_model('path')

    predictions = [ float(pred) for pred  in model.predict(input)[0] ]
    
    return { 
        'result': 1,
        'error': '',
        'data': predictions
    }
```

With the predict route's response, a graph displaying the probabilities for each prediction can be displayed. 

<center><img width="800" src="images/nn_predictions.png"></center>

## Deploying to Heroku

To deploy the developed code to Heroku, follow the steps below. 

1. Sign up for free and experience [Heroku](https://signup.heroku.com/login).

2. Now, it's time to create a new app. It is very important to connect the APP to our Github repository and enable the automatic deploys.

3. Install the Heroku CLI following the [instructions](https://devcenter.heroku.com/articles/heroku-cli).

4. Sign in to heroku using terminal

```bash
heroku login
```

5. In the root folder of the project check the heroku projects already created.

```bash
heroku apps
```

6. Check buildpack is correct: 

```bash
heroku buildpacks --app proj-mnist
```

7. Update the buildpack if necessary:

```bash
heroku buildpacks:set heroku/python --app proj-mnist
```

8. When you're running a script in an automated environment, you can [control Wandb with environment variables](https://docs.wandb.ai/guides/track/advanced/environment-variables) set before the script runs or within the script. Set up access to Wandb on Heroku, if using the CLI: 

```bash
heroku config:set WANDB_API_KEY=xxx --app proj-mnist
```

9. The instructions for launching an app are contained in a ``Procfile`` file that resides in the highest level of your project directory. Create the ``Procfile`` file with:

```bash
web: uvicorn source.api.main:app --host=0.0.0.0 --port=${PORT:-5000}
```

10. Configure the remote repository for Heroku:

```bash
heroku git:remote --app proj-mnist
```

11. Push all files to remote repository in Heroku. The command below will install all packages indicated in ``requirements.txt`` to Heroku VM. 

```bash
git push heroku main
```

12. Check the remote files run:

```bash
heroku run bash --app proj-mnist
```

13. If all previous steps were done with successful you will see the message below after open: ```https://proj-mnist.herokuapp.com/```.

14. For debug purposes whenever you can fetch your app’s most recent logs, use the [heroku logs command](https://devcenter.heroku.com/articles/logging#view-logs):

```bash
heroku logs
```

## Comparing the Classifer Results with a Decision Tree

A decision tree classifier was used to solve the same classification problem in a previous publication, which is available at [Using MNSIT Data to Train and Deploy a Classification Model](https://medium.com/@richardsonsantiago/using-mnsit-data-to-train-and-deploy-a-classification-model-6cfa1f1a5043).
This section investigates the differences between each model's predictions. 

The deployed application now includes a dropdown selector that allows the user to switch between classification models with ease. This demonstrates the neural network model's higher abstraction power, with its predictions succeeding where the decision tree's fails. 

<center><img width="800" src="images/index3.png"></center>

The MLP achieved an accuracy of 98 %, compared to the decision tree's 87 percent. The real impact of this 11% improvement can be seen in the confusion matrix comparison, where it is clear that the MLP's predictions are of a higher quality. 

The figure below depicts the differences in the confusion matrices obtained as a result of both models' classifications in the test set. When compared to the decision tree, the negative values in the last matrix indicate how many more samples were incorrectly classified by the decision tree model, while the positive values show a surplus of correct classifications by the neural network. 

<center><img width="800" src="images/comparison.png"></center> 

## References

- :book: Aurélien Géron. Hands on Machine Learning with Scikit-Learn, Keras and TensorFlow. [[Link]](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)