# Wind-Generated-Power

# Overview
This project is based on the **Wind-Generated-Power Prediction Hackathon by HackerEarth in 2021**. The project has been made using the **Pytorch** library. The data which was not available were assumed to be having the values of their corresponding column. I have achieved a score of **93.5** using this model uptill now.

# Various Sections
The project can be divided into following parts:
1. [Preparing the data](https://github.com/gandhisamay/Wind-Generated-Power/blob/main/Preparing%20data.py)
2. [Model](https://github.com/gandhisamay/Wind-Generated-Power/blob/main/model.py)
3. [Test Set Predictions](https://github.com/gandhisamay/Wind-Generated-Power/blob/main/PowerPredictionWeights.pt)

## 1. Preparing the data
For preparing the data, the Pandas Library has been used extensively, since it does the job of preparing the data in the easiest manner. One hot encoding has been used to replace the text data in the dataset. The characters and their onehotencods can be found [here](https://github.com/gandhisamay/Wind-Generated-Power/blob/main/Onehotencods.txt). As mentioned earlier all the missing data has been replaced by the median values of their corresponding column.

For preparing the dataset, I have used the ``Dataset`` class imported from ``torch.utils.data`` class provided in Pytorch. Since the data is raw so, I have done the custom implementation of the same which can be found [here](https://github.com/gandhisamay/Wind-Generated-Power/blob/main/dataset%20class.py).

Further ``DataLoader`` provided in ``torch.utils.data`` have been used while training the model.

## 2. Model
The model has been made by subclassing the ``nn.Module`` class provided in Pytorch. The model consists of 6 layers with 5 ReLU activation layers and 1 Softplus layer. I have trained the model for around 1500 epochs in total and to evaluate the model is not overfitting a dev set has been developed from the train dataset. The graph of loss vs iterations can also be found in above.

## 3. Test Set Predictions
The test set predictions can then be made and saved in a csv file.

# License
Everyone is free to use it if right credits are given 
