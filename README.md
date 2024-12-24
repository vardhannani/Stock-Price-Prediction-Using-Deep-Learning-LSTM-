# Stock-Price-Prediction-Using-Deep-Learning-LSTM-
Demonstrates how to predict stock prices using a Long Short-Term Memory (LSTM)
indetail analysis of how a nueral-network get's structured. 
1. Long Short-Term Memory (LSTM)
The core of this project is the LSTM model, which is a specialized type of Recurrent Neural Network (RNN). RNNs are particularly effective in handling sequences of data, like time series data (which stock prices are). However, traditional RNNs struggle to learn long-term dependencies due to the vanishing gradient problem. LSTM networks overcome this by maintaining a memory cell that preserves information over time.

The LSTM model used in this project is composed of several layers:

1. Embedding Layer: Transforms input into dense vectors of fixed size.

2. LSTM Layers: Learn patterns in sequential data.

3. Dropout Layers: Prevent overfitting by randomly setting a fraction of input units to zero.

4. Dense Layer: Outputs the predicted stock price.

This model follows the following process:

1. Data Collection: Stock data is downloaded using the Yahoo Finance API (yfinance).

2. Data Preprocessing: The stock data is normalized using MinMaxScaler to scale the values between 0 and 1.Time series data is split into sequences of 60 days to predict the next day's stock price.

3. Model Architecture: The model consists of two LSTM layers with 100 units each, followed by a dense layer for the output prediction.
   
4. Training: The model is trained using the Mean Squared Error (MSE) loss function and Adam optimizer.
   
5. Evaluation: The model's performance is evaluated by comparing predicted prices with actual prices using visualizations.
   
7. Prediction: Once trained, the model can predict stock prices for future days.

What Is the Vanishing Gradient Problem?

RNNs are trained using a process called Backpropagation Through Time (BPTT), which is an extension of the standard backpropagation algorithm used for training feedforward neural networks. The idea is to update the weights of the network by calculating the gradients of the loss function with respect to the weights, starting from the output layer and working backward through each time step.
RNNs are trained using a process called Backpropagation Through Time (BPTT), which is an extension of the standard backpropagation algorithm used for training feedforward neural networks. The idea is to update the weights of the network by calculating the gradients of the loss function with respect to the weights, starting from the output layer and working backward through each time step.
In traditional RNNs, as the gradients are propagated back through multiple time steps, they often get smaller and smaller. This happens because of the repeated multiplication of small values.
When these small gradients are backpropagated through many layers , they shrink exponentially. As a result, the weights associated with earlier time steps get updated very little or not at all, and the model cannot effectively learn long-term dependencies.

This is the my kind of explanation for better understanding.!

# Key Components of the Model
1. LSTM: LSTM units are specialized for learning sequential data and are ideal for tasks like stock price prediction, where the output depends on past values.
2. Dropout: Dropout layers help in regularization to avoid overfitting by randomly setting a fraction of the input units to zero during training.
3. Dense Layer: The final layer of the network generates the predicted output (next day’s stock price).

# Requirements
following libraries requried:

1. TensorFlow: For building and training the LSTM model.
2. NumPy: For numerical operations.
3. Pandas: For data manipulation.
4. Matplotlib: For plotting graphs.
5. Scikit-learn: For preprocessing and evaluation.
6. yfinance: For downloading stock data from Yahoo Finance.

# Evaluation

The performance of the model is evaluated by comparing the predicted stock prices with the actual stock prices. A plot of actual vs predicted stock prices is generated to visualize the model’s accuracy.
