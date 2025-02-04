This program (ga.cpp) is using the galib library to optimize the hyperparameters of the neural network defined in "nn.cpp."
"nn.cpp" contains a model of a neural net to predict the body fat percentage based on other low-cost measurements (features).
The GA is trying to use different parameters for the number of neurons in the first layer H1, the second layer H2, and the third layer H3.
The results of the GA are the recommended parameters to maximize the accuracy of prediction.

To use Mlpack please download and build as per instructions:https://www.mlpack.org/doc/quickstart/cpp.html#installing-mlpack
apt install might not work, bulding from source might be required

Please compile using this command:
g++ -std=c++14 -I/usr/local/include/gali -o ga ga.cpp nn.cpp -L/usr/local/lib -lga -larmadillo -fpermissive

to run: ./ga

Sample/Best output:
Best solution found:
H1: 48
H2: 64
H3: 48
Mean Squared Error on Prediction data points: 0.00105858