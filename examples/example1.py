import ml_framework as ml
import numpy as np

rng = np.random.default_rng(42)

model = ml.Sequential([
    ml.LinearLayer(in_features=256, out_features=32, rng=rng),
    ml.ActivationLayer(func=ml.ReLU, derivative=ml.ReLUDerivative),
    ml.Dropout(p=0.4, rng=rng),
    ml.LinearLayer(in_features=32, out_features=1, rng=rng)
])

optimizer = ml.Adam(learning_rate=0.001)
loss_fn = ml.BinaryCrossEntropyLoss()

print(model)