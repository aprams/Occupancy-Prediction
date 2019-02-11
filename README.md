# Small Data Classification

This project predicts occupancy on historical data for a sample dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For training a new network you will need python3 and for the inference server a Dockerfile is provided. Running that on your own is possible, but Docker is recommended.


### Installing

In order to train your own networks and start developing, you will need to install the requirements, preferably in a virtual environment:


```
virtualenv -p python3.6 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training a model can be done with an iPython notebook, you can run it using
```
jupyter notebook
```
and selecting the notebook.

## Deployment

Run the docker image with:

```
docker build -t occupancy-server . && docker run -p 5000:5000 occupancy-server:latest
```

You can then either use the 'upload_predict' endpoint at:
```
http://<DOCKER-IP>:5000/upload_predict
```

where you can upload a csv file or use curl on that endpoint with a POST request.
The result is a json containing the predictions for the next 24h per device.

## Built With

* [Docker](https://www.docker.com/) - Container Framework
* [Flask](http://flask.pocoo.org/) - Slim Web Framework
* [Keras](https://keras.io/) - High-level Deep Learning framework
* [Tensorflow](https://www.tensorflow.org/) - Deep Learning backend for Keras

## Authors

* **Alexander Prams** - [aprams](https://github.com/aprams)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Tensorflow authors for their awesome framework and the Dockerfile template


