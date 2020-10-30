# Yelp_Review_Recommendation_Score_Prediction
Predict the Recommendation score of yelp review.

Data url : https://www.kaggle.com/ayushjain601/yelp-reviews-dataset   (only test dataset containing 50k review samples is used.)

total no. of review samples = 50k

Train-Test split : 40k : 10k

Input : review-text       Output : Recommendation score (rating) {1,2,3,4,5} or {0,1,2,3,4}



Model : LSTM-DENSE..

![lstm_perceptron_performance.png](https://github.com/shashi-cs17/Yelp_Review_Recommendation_Score_Prediction/blob/v0.0_/performance/lstm_perceptron_performance.png)

Model : LSTM-CNN-MLP...

![lstm_cnn_mlp.png](https://github.com/shashi-cs17/Yelp_Review_Recommendation_Score_Prediction/blob/v0.0_/performance/lstm_cnn_mlp.png)

Model : CNN-VADER-MLP...

![vader_cnn.png](https://github.com/shashi-cs17/Yelp_Review_Recommendation_Score_Prediction/blob/v0.0_/performance/vader_cnn.png)

Model : LSTM-VADER-MLP..

![vader_lstm_mlp.png](https://github.com/shashi-cs17/Yelp_Review_Recommendation_Score_Prediction/blob/v0.0_/performance/vader_lstm_mlp.png)

Model : LSTM-slidingWindowVADER-MLP..

![lstm_window-vader_mlp.png](https://github.com/shashi-cs17/Yelp_Review_Recommendation_Score_Prediction/blob/v0.0_/performance/lstm_window-vader_mlp.png)



