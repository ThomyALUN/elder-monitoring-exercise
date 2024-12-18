import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from minisom import MiniSom
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Semilla para reproducibilidad
seed = 0
np.random.seed(seed)

# 1. Cargar el dataset
single_elder_home_monitoring_gas_and_position = fetch_ucirepo(id=799)
X = single_elder_home_monitoring_gas_and_position.data.features
y = single_elder_home_monitoring_gas_and_position.data.targets
data = pd.concat([X, y], axis=1)

# 2. Preprocesamiento de los datos
# Selección de las columnas relevantes
features = [
    'temperature', 'humidity', 'CO2CosIRValue', 'CO2MG811Value',
    'MOX1', 'MOX2', 'MOX3', 'MOX4', 'COValue'
]
data_features = data[features]

# Normalización de las características
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_features)

# PCA para corrección ambiental, igual a lo utilizado en el artículo
pca = PCA(n_components=2)  # Reducir dimensiones para capturar varianza ambiental
data_pca = pca.fit_transform(data_normalized)
print(f"Varianza explicada por PCA: {pca.explained_variance_ratio_}")

# Datos finales para SOM después de eliminar variabilidad ambiental
data_som = data_normalized - pca.inverse_transform(data_pca)

# 3. Configuración del SOM
som_grid_x = 2  
som_grid_y = 2
som = MiniSom(som_grid_x, som_grid_y, data_som.shape[1], sigma=1.0, learning_rate=0.5, random_seed=seed)

# Inicializar pesos del SOM
som.random_weights_init(data_som)

# Entrenamiento del SOM
print("Entrenando el SOM...")
som.train_random(data_som, num_iteration=1000)

# Crear la carpeta /plots si no existe
if not os.path.exists('plots'):
    os.makedirs('plots')

# 4. Visualización de resultados
# Mapas de hits: Número de muestras asignadas a cada nodo
plt.figure(figsize=(10, 8))
plt.title('Mapa de Hits')
sns.heatmap(som.activation_response(data_som), cmap='coolwarm')
plt.savefig('plots/mapa_de_hits.png')
plt.close()

for i, feature in enumerate(features):
    plt.figure(figsize=(8, 6))
    plt.title(f"Mapa de {feature}")
    sns.heatmap(som.get_weights()[:, :, i], cmap='viridis', cbar=True)  # cbar=True incluye la barra de color
    plt.savefig(f'plots/mapa_de_{feature}.png')
    plt.close()

# 5. Clustering con SOM
# Asignar cada muestra a un nodo del SOM
winning_nodes = np.array([som.winner(x) for x in data_som])

# Generar nombres secuenciales usando el abecedario
alphabet = string.ascii_uppercase
cluster_names = {}
name_counter = 0

# Crear un diccionario para los nombres de los clusters
for x in range(som_grid_x):
    for y in range(som_grid_y):
        # Generar nombre (ejemplo: A, B, ... Z, AA, AB, ...)
        cluster_name = ""
        temp_counter = name_counter
        while True:
            cluster_name = alphabet[temp_counter % 26] + cluster_name
            temp_counter = temp_counter // 26 - 1
            if temp_counter < 0:
                break
        cluster_names[(x, y)] = cluster_name
        name_counter += 1

# Generar nombres secuenciales usando el abecedario
alphabet = string.ascii_uppercase
cluster_names = {}
name_counter = 0

# Crear un diccionario para los nombres de los clusters
for x in range(som_grid_x):
    for y in range(som_grid_y):
        # Generar nombre (ejemplo: A, B, ..., Z, AA, AB, ...)
        cluster_name = ""
        temp_counter = name_counter
        while True:
            cluster_name = alphabet[temp_counter % 26] + cluster_name
            temp_counter = temp_counter // 26 - 1
            if temp_counter < 0:
                break
        cluster_names[(x, y)] = cluster_name
        name_counter += 1

# Obtener asignaciones de clusters para cada instancia
class_assignments = np.array([som.winner(x) for x in data_som])

# Agregar una columna al DataFrame con el nombre del cluster
data['Cluster Name'] = [cluster_names.get(tuple(cluster), "Unknown") for cluster in class_assignments]

# Guardar el DataFrame con la nueva columna si lo necesitas
data.to_csv('data_with_clusters.csv', index=False)
print("Datos guardados en 'data_with_clusters.csv'")