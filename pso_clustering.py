"""
Clustering con Particle Swarm Optimization (PSO) — dataset
Student Performance Factors (Kaggle).

Curso: Aprendizaje de Máquina - UNAP
Actividad 03 - Parte: Cristian

IDEA DEL ALGORITMO:
En K-Means, los centroides se ajustan iterativamente promediando
los puntos de cada cluster. Aquí en cambio usamos PSO: cada
partícula del enjambre representa un conjunto COMPLETO de K
centroides (un vector de tamaño K x n_features), y el PSO mueve
ese vector en el espacio buscando la posición que minimiza la
inercia (suma de distancias al cuadrado de cada punto a su
centroide más cercano) — el mismo objetivo que optimiza K-Means,
pero resuelto con enjambre en vez de con el algoritmo clásico.

Al final comparamos el resultado del PSO contra K-Means de
sklearn, para ver qué tan bien le fue al enjambre frente al
método "de fábrica".
"""

import os
import glob
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. DATOS — Student Performance Factors (Kaggle)
# ---------------------------------------------------------------------
path = kagglehub.dataset_download(
    "mosapabdelghany/student-performance-factors-dataset"
)
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

# Para clustering no usamos Exam_Score como feature (lo dejamos
# aparte solo para interpretar los clusters al final); se agrupa
# a los estudiantes por sus características, no por su nota.
exam_score = df["Exam_Score"].values.astype(float)
X_df = df.drop(columns=["Exam_Score"])

num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

scaler = StandardScaler()
X_num = scaler.fit_transform(X_df[num_cols]) if num_cols else np.empty((len(df), 0))

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(X_df[cat_cols]) if cat_cols else np.empty((len(df), 0))

X = np.hstack([X_num, X_cat])

N_SAMPLES, N_FEATURES = X.shape
K = 3   # número de clusters (Bajo / Medio / Alto rendimiento, por ejemplo)

# ---------------------------------------------------------------------
# 2. FUNCIÓN DE APTITUD (inercia, igual que K-Means)
# ---------------------------------------------------------------------
def unpack_centroids(theta):
    """Convierte el vector plano de una partícula en una matriz
    de K centroides x N_FEATURES."""
    return theta.reshape(K, N_FEATURES)


def asignar_clusters(centroides, X):
    """Para cada punto, calcula el centroide más cercano.
    Devuelve el índice de cluster asignado a cada fila de X."""
    distancias = np.linalg.norm(
        X[:, None, :] - centroides[None, :, :], axis=2
    )  # (N_SAMPLES, K)
    return np.argmin(distancias, axis=1), distancias


def fitness(theta, X):
    """Inercia: suma de distancias al cuadrado de cada punto a su
    centroide asignado (a minimizar, igual que el objetivo interno
    de K-Means)."""
    centroides = unpack_centroids(theta)
    etiquetas, distancias = asignar_clusters(centroides, X)
    dist_min = distancias[np.arange(len(X)), etiquetas]
    return np.sum(dist_min ** 2)


N_PARAMS = K * N_FEATURES

# ---------------------------------------------------------------------
# 3. PSO — CICLO DEL ALGORITMO DE ENJAMBRE
# ---------------------------------------------------------------------
def pso_clustering(n_particles=30, n_iter=100, w=0.72, c1=1.5, c2=1.5,
                    v_max=0.5, seed=42):
    """
    1) Representación de la partícula: vector theta en R^(K x n_features)
       con las coordenadas de los K centroides.
    2) Inicialización: cada partícula arranca en K puntos reales del
       dataset elegidos al azar (mejor que puntos totalmente
       aleatorios, ayuda a converger más rápido).
    3) Función de aptitud: inercia (fitness()).
    4) Comportamiento de la partícula: combina inercia (w), pbest
       y gbest para actualizar velocidad y posición.
    5) Evolución: se repite iterativamente, actualizando pbest/gbest.
    6) Finalización: se detiene al llegar a n_iter iteraciones.
    """
    rng = np.random.default_rng(seed)

    positions = np.zeros((n_particles, N_PARAMS))
    for i in range(n_particles):
        idx_iniciales = rng.choice(N_SAMPLES, size=K, replace=False)
        positions[i] = X[idx_iniciales].ravel()

    velocities = rng.uniform(-0.1, 0.1, size=(n_particles, N_PARAMS))

    pbest_pos = positions.copy()
    pbest_val = np.array([fitness(p, X) for p in positions])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [gbest_val]

    for it in range(n_iter):
        r1 = rng.random((n_particles, N_PARAMS))
        r2 = rng.random((n_particles, N_PARAMS))

        cognitive = c1 * r1 * (pbest_pos - positions)
        social = c2 * r2 * (gbest_pos - positions)
        velocities = w * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities

        values = np.array([fitness(p, X) for p in positions])

        improved = values < pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = values[improved]

        best_idx = np.argmin(pbest_val)
        if pbest_val[best_idx] < gbest_val:
            gbest_val = pbest_val[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()

        history.append(gbest_val)

        if (it + 1) % 20 == 0:
            print(f"Iteración {it + 1:3d}/{n_iter} - "
                  f"mejor inercia (gbest): {gbest_val:.2f}")

    return gbest_pos, gbest_val, history


if __name__ == "__main__":
    print(f"Filas: {N_SAMPLES} | Features tras preprocesamiento: {N_FEATURES}")
    print(f"Parámetros a optimizar por partícula (K x features): {N_PARAMS}")
    print(f"Número de clusters (K): {K}")
    print("\nEjecutando clustering con PSO sobre Student Performance...\n")

    best_theta, best_inertia, history = pso_clustering()
    centroides_pso = unpack_centroids(best_theta)
    etiquetas_pso, _ = asignar_clusters(centroides_pso, X)

    sil_pso = silhouette_score(X, etiquetas_pso)

    print("\n--- Resultado PSO ---")
    print(f"Inercia final: {best_inertia:.2f}")
    print(f"Silhouette score: {sil_pso:.4f}")
    print("Tamaño de cada cluster:",
          {i: int(np.sum(etiquetas_pso == i)) for i in range(K)})

    # -------------------------------------------------------------
    # Comparación contra K-Means (línea base "de fábrica")
    # -------------------------------------------------------------
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    etiquetas_kmeans = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, etiquetas_kmeans)

    print("\n--- Comparación PSO vs. K-Means ---")
    print(f"Inercia PSO:      {best_inertia:.2f} | Silhouette PSO:      {sil_pso:.4f}")
    print(f"Inercia K-Means:  {kmeans.inertia_:.2f} | Silhouette K-Means:  {sil_kmeans:.4f}")

    # -------------------------------------------------------------
    # Interpretación: promedio de Exam_Score por cluster (PSO)
    # -------------------------------------------------------------
    print("\n--- Exam_Score promedio por cluster (PSO) ---")
    for c in range(K):
        promedio = exam_score[etiquetas_pso == c].mean()
        print(f"Cluster {c}: Exam_Score promedio = {promedio:.2f} "
              f"(n={int(np.sum(etiquetas_pso == c))})")

    # -------------------------------------------------------------
    # Gráficos
    # -------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteración")
    plt.ylabel("Mejor inercia global (gbest)")
    plt.title("Convergencia de PSO — Clustering Student Performance")
    plt.tight_layout()
    plt.savefig("convergencia_pso_clustering.png", dpi=150)
    print("\nGráfico de convergencia guardado en convergencia_pso_clustering.png")

    # Visualización en 2D con PCA (el dataset tiene muchas más
    # dimensiones tras el one-hot, así que reducimos para poder verlo)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    centroides_2d = pca.transform(centroides_pso)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=etiquetas_pso,
                           cmap="viridis", alpha=0.5, s=15)
    plt.scatter(centroides_2d[:, 0], centroides_2d[:, 1],
                c="red", marker="X", s=200, edgecolors="black",
                label="Centroides (PSO)")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title("Clusters encontrados por PSO (proyección PCA 2D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("clusters_pso_pca.png", dpi=150)
    print("Gráfico de clusters (PCA 2D) guardado en clusters_pso_pca.png")