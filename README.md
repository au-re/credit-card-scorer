# Application of an Artificial Neural Network to Assess Credit Risk

Laboratory Project for the course 'Intelligent Data Analysis Applications in Business' 2015-2016 at the UPC Barcelona

The following code attempts to assess credit card risk based on user data. The 
data set is analyzed with a neural network. The library used to implement the 
ANN is google's [tensorflow](https://www.tensorflow.org/).

## How to run this code

You will need python 3 to run this code. You will also need to install some dependencies,
run the following commands in your terminal:

`pip install scikit-learn`

`pip install numpy`

`sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl`

Once the dependencies are installed navigate to the folder 'src' and start the 
ANN with:

`python ccscorer.py`

This will run the neural network on the dataset. 
