# --------------------------
# IMPORTS
# --------------------------
import pandas as pd
from collections import deque
from river import linear_model, tree, neighbors, ensemble, metrics, preprocessing, compose, naive_bayes, optim, base

# --------------------------
# CUSTOM LAG TRANSFORMER
# --------------------------
class LagFeatures(base.Transformer):
    """River-compatible transformer for lag features with buffer injection."""
    def __init__(self, lags=3, max_history=1000, buffer=None):
        self.lags = lags
        self.max_history = max_history
        # allow dependency injection of storage (list, deque, Redis mock)
        self.buffer = buffer if buffer is not None else deque(maxlen=max_history)

    def learn_one(self, x, y=None):
        self.buffer.append(y)
        return self

    def transform_one(self, x):
        features = {}
        history_list = list(self.buffer)
        for i, val in enumerate(reversed(history_list[-self.lags:]), 1):
            features[f'lag_{i}'] = 0 if val is None else val
        for i in range(1, self.lags + 1):
            if f'lag_{i}' not in features:
                features[f'lag_{i}'] = 0
        return features

# --------------------------
# LOAD DATA
# --------------------------
df = pd.read_csv('air.csv')  # your CSV file
target_col = 'passengers'
horizon = 12

train_df = df.iloc[:-horizon]
val_df = df.iloc[-horizon:]

# --------------------------
# MODELS TO TEST
# --------------------------
models_to_test = [
    compose.Pipeline(
        ('lag_features', LagFeatures(lags=3)),
        ('scale', preprocessing.StandardScaler()),
        ('model', linear_model.LinearRegression(optimizer=optim.SGD(lr=0.001), l2=0.01))
    ),
    compose.Pipeline(
        ('lag_features', LagFeatures(lags=3)),
        ('scale', preprocessing.StandardScaler()),
        ('model', linear_model.BayesianLinearRegression(alpha=1, beta=1))
    ),
    linear_model.PARegressor(C=0.01, mode=2),
    tree.HoeffdingTreeRegressor(grace_period=500, leaf_prediction='adaptive'),
    tree.HoeffdingAdaptiveTreeRegressor(grace_period=500),
    ensemble.BaggingRegressor(model=tree.HoeffdingTreeRegressor(), n_models=3),
    naive_bayes.GaussianNB(),
    neighbors.KNNRegressor(n_neighbors=5)
]

# --------------------------
# EVALUATION
# --------------------------
results = {}
for model in models_to_test:
    model_name = type(model).__name__ if not isinstance(model, compose.Pipeline) else type(model['model']).__name__
    lag_transformer = LagFeatures(lags=3)

    # initialize metrics
    mae = metrics.MAE()
    mape = metrics.MAPE()
    rmse = metrics.RMSE()
    r2 = metrics.R2()

    # TRAIN & ONLINE EVALUATION
    for _, row in train_df.iterrows():
        x = {}
        y = row[target_col]
        lag_transformer.learn_one(x, y)
        features = lag_transformer.transform_one(x)

        # predict before learning (online)
        y_pred = model.predict_one(features)
        if y_pred is not None:
            mae.update(y, y_pred)
            mape.update(y, y_pred)
            rmse.update(y, y_pred)
            r2.update(y, y_pred)

        # learn from current sample
        model.learn_one(features, y)

    # VALIDATE
    for _, row in val_df.iterrows():
        x = {}
        y = row[target_col]
        lag_transformer.learn_one(x, y)
        features = lag_transformer.transform_one(x)
        y_pred = model.predict_one(features)
        if y_pred is not None:
            mae.update(y, y_pred)
            mape.update(y, y_pred)
            rmse.update(y, y_pred)
            r2.update(y, y_pred)
        model.learn_one(features, y)

    results[model_name] = {
        "MAE": mae.get(),
        "MAPE": mape.get(),
        "RMSE": rmse.get(),
        "R2": r2.get()
    }

# --------------------------
# PRINT RESULTS
# --------------------------
print("=== Benchmark Results ===")
for name, metrics_dict in results.items():
    print(f"{name}: ", end="")
    print(", ".join([f"{k}={v:.2f}" for k, v in metrics_dict.items()]))

# --------------------------
# ONLINE LEARNING SIMULATION EXAMPLE
# --------------------------
print("\n=== Online Learning Simulation with KNN ===")
lag_transformer = LagFeatures(lags=3)
online_model = compose.Pipeline(
    ('lag_features', lag_transformer),
    ('scale', preprocessing.StandardScaler()),  # optional for KNN
    ('model', neighbors.KNNRegressor(n_neighbors=5))
)

# metrics for online simulation
online_mae = metrics.MAE()
online_mape = metrics.MAPE()
online_rmse = metrics.RMSE()
online_r2 = metrics.R2()

for i, row in df.iterrows():
    x = {}
    y = row[target_col]
    lag_transformer.learn_one(x, y)
    features = lag_transformer.transform_one(x)
    y_pred = online_model.predict_one(features)
    if y_pred is not None:
        online_mae.update(y, y_pred)
        online_mape.update(y, y_pred)
        online_rmse.update(y, y_pred)
        online_r2.update(y, y_pred)
    online_model.learn_one(features, y)

    if i < 5:
        print(f"Step {i}, True={y}, Predicted={y_pred}")

print(f"Online learning MAE with KNN after full pass: {online_mae.get():.2f}")
print(f"Online learning MAPE with KNN: {online_mape.get():.2f}")
print(f"Online learning RMSE with KNN: {online_rmse.get():.2f}")
print(f"Online learning R2 with KNN: {online_r2.get():.2f}")
