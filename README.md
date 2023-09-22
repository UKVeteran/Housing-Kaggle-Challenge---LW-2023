# Housing-Kaggle-Challenge - Regression Techniques
![kaggle_5407_media_housesbanner](https://github.com/UKVeteran/Housing-Kaggle-Challenge---LW-2023/assets/39216339/a556ebb9-190a-48aa-b083-566cce1fc8be)

# RMSE

The root-mean-square error (RMSE) is a measure used to assess how well a predictive model, such as a machine learning algorithm, is performing. 
It is a way to quantify the average difference between the predicted values and the actual (observed) values. Here's a simple explanation:

1 - Squared Differences: For each data point, you calculate the squared difference between the predicted value and the actual value. This is done to make sure that both overestimations and underestimations contribute to the error, without canceling each other out. <br>
2 - Average: You then take the average (mean) of all these squared differences. This gives you a measure of the typical error the model makes across all data points.<br>
3 - Square Root: Finally, you take the square root of this average to get the RMSE. This step is important because it ensures that the RMSE is in the same units as the original data, making it easier to interpret.

In simpler terms, RMSE tells you how far off, on average, our model's predictions are from the actual values. Smaller RMSE values indicate that the model's predictions are closer to the actual values, while larger RMSE values mean the predictions are further away. 
It is a way to quantify the goodness of fit of your model to the data, with lower values indicating a better fit.

# Stack'em Up - RMSE Score = 0.1204
```python
catboost = CatBoostRegressor()
xgboost = XGBRegressor(max_depth=10, n_estimators=300, learning_rate=0.05)
gboost = GradientBoostingRegressor(n_estimators=100)
ridge = Ridge(alpha= 0.9736842105263157)
SVM = SVR(C=1, epsilon=0.05)
adaboost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None))

model = StackingRegressor(
    estimators=[("gboost", gboost),("adaboost", adaboost),("ridge", ridge), 
                ("svm", SVM), ("cat", catboost), ("XGB", xgboost)],
    
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

pipe_stacking = make_pipeline(preproc, model)
score = cross_val_score(pipe_stacking, X, y_log, cv=5, scoring=rmse, n_jobs=-1)
print(score.std())
score.mean()
```
