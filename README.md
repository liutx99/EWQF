# EWQF
Ensemble Water Quality Forecasting based on Decomposition, Sub-model Selection, and Adaptive Interval

In this document, it is divided into two main sections.

The first part is the data decomposition, which is in the IVMD folder. The steps to run this part of the code are as follows:
1. Run IGWO_VMD to find the optimal parameters for VMD.
2. Run VMD_sliding to decompose the data.

The second part is the integration point prediction and this part of the code is found in the main folder. The steps to run this part of the code are as follows:
1. Run STEP1: Calculate the performance of each component on the validation set and record the evaluation metrics (MAE, RMSE, STD, MAPE and R2)
2. Run STEP2: Normalise the evaluation metrics of each model for each component, calculate the CEI value of the model on each component, and select the model corresponding to the smallest CEI as the optimal sub-model.
3. Run STEP3: Predict the component with the optimal sub-model and finally evaluate its performance on the test set.

