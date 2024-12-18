import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Evitar desbordamiento numérico
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def initialize_weights(layer_sizes, seed):
    np.random.seed(seed)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
        b = np.zeros((1, layer_sizes[i + 1]))
        weights.append(W)
        biases.append(b)
    return weights, biases

def forward_pass(X, weights, biases, layers):
    activations = [X]
    Z_values = []
    for i in range(len(layers) - 2):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        A = relu(Z)
        Z_values.append(Z)
        activations.append(A)
    Z = np.dot(activations[-1], weights[-1]) + biases[-1]
    A = softmax(Z)
    activations.append(A)
    Z_values.append(Z)
    return activations, Z_values

def backward_pass(y_true, activations, Z_values, weights):
    grads_w = []
    grads_b = []
    dA = -(y_true - activations[-1])
    for i in reversed(range(len(weights))):
        dZ = dA if i == len(weights) - 1 else dA * relu_derivative(Z_values[i])
        dW = np.dot(activations[i].T, dZ) / len(y_true)
        db = np.sum(dZ, axis=0, keepdims=True) / len(y_true)
        dA = np.dot(dZ, weights[i].T)
        grads_w.insert(0, dW)
        grads_b.insert(0, db)
    return grads_w, grads_b

def evaluate_nn(X, y_true, weights, biases, layers):
    activations, _ = forward_pass(X, weights, biases, layers)
    loss = mse_loss(y_true, activations[-1])
    return loss

def train_nn(X_train, y_train, X_test, y_test, layer_sizes, learning_rate, epochs, seed):
    weights, biases = initialize_weights(layer_sizes, seed)
    mse_history_train = []
    mse_history_test = []
    for epoch in range(epochs):
        activations, Z_values = forward_pass(X_train, weights, biases, layer_sizes)
        loss_train = mse_loss(y_train, activations[-1])
        grads_w, grads_b = backward_pass(y_train, activations, Z_values, weights)
        for i in range(len(weights)):
            weights[i] -= learning_rate * grads_w[i]
            biases[i] -= learning_rate * grads_b[i]
        loss_test = evaluate_nn(X_test, y_test, weights, biases, layer_sizes)
        mse_history_train.append(loss_train)
        mse_history_test.append(loss_test)
        if (epoch + 1) % 50 == 0:
            print(f"Seed {seed}, Epoch {epoch+1}/{epochs} - Train Loss: {loss_train:.4f} - Test Loss: {loss_test:.4f}")
    return mse_history_train, mse_history_test, weights, biases
# Configuración de hiperparámetros
seeds = [42, 123]
learning_rates = [0.01, 0.1, 0.5]
layer_configs = [
    [9, 4],      
    [9, 16, 4],
    [9, 32, 4],
    [9, 16, 8, 4],
    [9, 32, 16, 4], 
]
results = []
# Crear un DataFrame con los datos
data = pd.read_csv('data_with_clusters.csv').drop('timestamp', axis=1)
X = data.drop('Cluster Name', axis=1)
y = data['Cluster Name']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = one_hot_encode(y_encoded, num_classes=4)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
all_mse_histories = {}
# Entrenar redes neuronales
for seed in seeds:
    for lr in learning_rates:
        for layers in layer_configs:
            print(f"Entrenando con Seed={seed}, Learning Rate={lr}, Layers={layers}")
            mse_history_train, mse_history_test, _, _ = train_nn(X_train, y_train, X_test, y_test, layers, lr, epochs=10000, seed=seed)
            key = f"Seed={seed}, LR={lr}, Layers={len(layers)-1}"
            all_mse_histories[key] = mse_history_train
            results.append({
                'Seed': seed,
                'Learning Rate': lr,
                'Layers': len(layers) - 1,
                'Final Train MSE': mse_history_train[-1],
                'Final Test MSE': mse_history_test[-1]
            })
# Crear DataFrame y exportar resultados a Excel
df_results = pd.DataFrame(results)
df_results.to_excel("nn_comparison_results.xlsx", index=False)
print("Resultados guardados en 'nn_comparison_results.xlsx'")
# Graficar comparativa
plt.figure(figsize=(10, 6))
for key, mse_history in all_mse_histories.items():
    plt.plot(mse_history, label=key)
plt.title('Comparación de Error Cuadrático Medio')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.show()
