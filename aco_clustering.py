import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay

# Silenciar advertencias secundarias
warnings.filterwarnings("ignore")

# Estilos de consola ANSI para formateo en tiempo real
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"


# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS REALES (WINE RECOGNITION DATASET)
# =============================================================================
# Carga del dataset real benchmark desde scikit-learn
dataset_raw = load_wine()

# Definición explícita de las 13 columnas de características químicas (features)
feature_cols = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline"
]

# Construcción de DataFrame para manipulación estructurada de los datos
df_X = pd.DataFrame(dataset_raw.data, columns=feature_cols)
df_y = pd.Series(dataset_raw.target, name="tipo_vino")

# Nombres de las clases reales / cultivos de vino
class_names = ["Cultivo 1", "Cultivo 2", "Cultivo 3"]
nombre_dataset = "Wine Recognition Benchmark Dataset (UCI)"

# Escalamiento estandarizado de características (media=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_X)
n_muestras, n_caracteristicas = X_scaled.shape


# =============================================================================
# 2. DEFINICIÓN DE PARÁMETROS Y VARIABLES DEL ALGORITMO ACO CLÁSICO
# =============================================================================
N_CLUSTERS = 3             # Número de grupos / centroides a hallar (K=3)
NUM_HORMIGAS = 15          # Número de hormigas exploradoras en la colonia
MAX_ITERACIONES = 20       # Número de iteraciones de la colonia ACO

# Parámetros del ACO Clásico Discreto
ALPHA = 1.0                # Peso relativo de la feromona histórica (α)
BETA = 2.0                 # Peso relativo de la información heurística (β)
RHO = 0.1                  # Tasa de evaporación de feromonas (ρ = 10%)
Q_DEPOT = 1000.0           # Constante de depósito de feromona (Q)
TAU_0 = 1.0                # Feromona inicial implícita (τ_0)


# =============================================================================
# 3. ESTRUCTURA, FUNCIONES Y CONTROLADOR DE ACO CLÁSICO
# =============================================================================
class HormigaClasica:
    def __init__(self, n_muestras):
        self.asignacion = np.zeros(n_muestras, dtype=int)  # Vector S discreto
        self.centroides = None                             # Matriz K x n_features
        self.sse = np.inf                                  # Suma de errores al cuadrado

    def construir_solucion(self, probabilidades_matriz, X_datos):
        N = X_datos.shape[0]
        self.asignacion = np.zeros(N, dtype=int)
        
        # Cada observacion 'i' selecciona su clúster 'k' según las probabilidades calculadas
        for i in range(N):
            self.asignacion[i] = np.random.choice(N_CLUSTERS, p=probabilidades_matriz[i])
            
        # Re-calcular centroides C_k como el centro de masa de las observaciones asignadas
        self.centroides = np.zeros((N_CLUSTERS, n_caracteristicas))
        for k in range(N_CLUSTERS):
            mascara = (self.asignacion == k)
            if np.sum(mascara) > 0:
                self.centroides[k] = X_datos[mascara].mean(axis=0)
            else:
                self.centroides[k] = X_datos[np.random.choice(N)]

        dists = np.zeros((N, N_CLUSTERS))
        for k in range(N_CLUSTERS):
            dists[:, k] = np.linalg.norm(X_datos - self.centroides[k], axis=1)
        self.asignacion = np.argmin(dists, axis=1)

        self.sse = 0.0
        for k in range(N_CLUSTERS):
            mascara = (self.asignacion == k)
            if np.sum(mascara) > 0:
                self.centroides[k] = X_datos[mascara].mean(axis=0)
                self.sse += float(np.sum((X_datos[mascara] - self.centroides[k]) ** 2))


class OptimizadorACOClasico:
    def __init__(self, num_hormigas=NUM_HORMIGAS, max_iter=MAX_ITERACIONES,
                 alpha=ALPHA, beta=BETA, rho=RHO, q=Q_DEPOT):
        self.num_hormigas = num_hormigas
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        
        # MATRIZ REAL DE FEROMONAS τ_{i,k} (Tamaño N x K)
        # Inicializada con feromona base τ_0 = 1.0
        self.matriz_feromonas = np.ones((n_muestras, N_CLUSTERS)) * TAU_0
        
        # Centroides globales de referencia para la heurística
        self.centroides_actuales = X_scaled[np.random.choice(n_muestras, N_CLUSTERS, replace=False)].copy()
        
        self.gbest_asignacion = None
        self.gbest_centroides = None
        self.gbest_sse = np.inf
        
        self.historial_gbest_sse = []
        self.historial_promedio_sse = []
        self.registros_hormigas = []

    def calcular_matriz_heuristica(self, X_datos):
        diferencias_distancia = np.zeros((n_muestras, N_CLUSTERS))
        for k in range(N_CLUSTERS):
            diferencias_distancia[:, k] = np.linalg.norm(X_datos - self.centroides_actuales[k], axis=1)
        
        matriz_heuristica = 1.0 / (diferencias_distancia + 1e-6)
        return matriz_heuristica

    def calcular_probabilidades_transicion(self, matriz_heuristica):
        numeradores = (self.matriz_feromonas ** self.alpha) * (matriz_heuristica ** self.beta)
        probabilidades = numeradores / numeradores.sum(axis=1, keepdims=True)
        return probabilidades

    def optimizar(self, X_datos):
        print(f"\n{BOLD}{CYAN}==========================================================================")
        print(f"       OPTIMIZACIÓN CLÁSICA DISCRETA CON COLONIA DE HORMIGAS (ACO CLUSTERING)")
        print(f"=========================================================================={RESET}")
        print(f" Dataset Real          : {BOLD}{CYAN}{nombre_dataset}{RESET}")
        print(f" Tamaño Matriz Feromona: {BOLD}{n_muestras} observaciones × {N_CLUSTERS} clústeres{RESET} (τ_ik)")
        print(f" Tamaño de Colonia     : {BOLD}{self.num_hormigas}{RESET} hormigas discretas")
        print(f" Iteraciones Máximas   : {BOLD}{self.max_iter}{RESET} iteraciones")
        print(f" Parámetros ACO        : α={self.alpha}, β={self.beta}, Tasa Evaporación ρ={self.rho}, Depósito Q={self.q}\n")

        # Bucle principal de generaciones ACO
        for iteracion in range(1, self.max_iter + 1):
            print(f"\n{BOLD}{MAGENTA}┌─────────────────────────────────────────────────────────────────────────┐{RESET}")
            print(f"{BOLD}{MAGENTA}│ ITERACIÓN {iteracion:02d}/{self.max_iter:02d}  (Muestreo de Decisión P(i → k) ∝ τ_ik^α * η_ik^β)         │{RESET}")
            print(f"{BOLD}{MAGENTA}└─────────────────────────────────────────────────────────────────────────┘{RESET}")
            
            # 1. Calcular matriz heurística η_ik y probabilidades P(i → k)
            matriz_heuristica = self.calcular_matriz_heuristica(X_datos)
            probabilidades_matriz = self.calcular_probabilidades_transicion(matriz_heuristica)
            
            hormigas_iteracion = []
            scores_iteracion = []
            
            # 2. Construcción de soluciones de la colonia
            for h in range(self.num_hormigas):
                hormiga = HormigaClasica(n_muestras)
                hormiga.construir_solucion(probabilidades_matriz, X_datos)
                
                hormigas_iteracion.append(hormiga)
                scores_iteracion.append(hormiga.sse)
                
                tag_estado = ""
                if hormiga.sse < self.gbest_sse:
                    self.gbest_sse = hormiga.sse
                    self.gbest_asignacion = hormiga.asignacion.copy()
                    self.gbest_centroides = hormiga.centroides.copy()
                    tag_estado = f" {BOLD}{GREEN}[🔥 NUEVO BEST GLOBAL ACO! SSE = {hormiga.sse:.2f}]{RESET}"
                
                silueta_temp = float(silhouette_score(X_datos, hormiga.asignacion))
                
                self.registros_hormigas.append({
                    "iteracion": iteracion,
                    "hormiga": h + 1,
                    "sse": hormiga.sse,
                    "silueta": silueta_temp
                })
                
                simbolo = "└─" if h == self.num_hormigas - 1 else "├─"
                print(f"  {simbolo} Hormiga {h+1:02d}/{self.num_hormigas:02d} │ Inercia (SSE): {hormiga.sse:8.2f} │ Silueta: {silueta_temp:.4f}{tag_estado}")
                sys.stdout.flush()

            # Actualizar centroides globales de referencia
            self.centroides_actuales = self.gbest_centroides.copy()
            
            # 3. EVAPORACIÓN DE FEROMONAS: τ_ik ← (1 - ρ) * τ_ik
            self.matriz_feromonas *= (1.0 - self.rho)
            
            # 4. DEPÓSITO DE FEROMONAS POR LAS MEJORES HORMIGAS: Δτ = Q / SSE
            indices_top = np.argsort(scores_iteracion)[:3]
            for idx in indices_top:
                best_ant = hormigas_iteracion[idx]
                delta_tau = self.q / best_ant.sse
                for i in range(n_muestras):
                    cluster_elegido = best_ant.asignacion[i]
                    self.matriz_feromonas[i, cluster_elegido] += delta_tau

            promedio_sse_iter = float(np.mean(scores_iteracion))
            std_sse_iter = float(np.std(scores_iteracion))
            self.historial_gbest_sse.append(self.gbest_sse)
            self.historial_promedio_sse.append(promedio_sse_iter)
            
            silueta_gbest = float(silhouette_score(X_datos, self.gbest_asignacion))
            
            print(f"\n  {BOLD}{GREEN}▶ RESUMEN ITERACIÓN {iteracion:02d}/{self.max_iter:02d}:{RESET}")
            print(f"    • Mejor Inercia Global ACO (gbest SSE) : {BOLD}{GREEN}{self.gbest_sse:.2f}{RESET}")
            print(f"    • Coeficiente de Silueta ACO          : {BOLD}{CYAN}{silueta_gbest:.4f}{RESET}")
            print(f"    • Promedio Colonia ACO (SSE)          : {promedio_sse_iter:.2f} (± {std_sse_iter:.2f})")

        print(f"\n{BOLD}{GREEN}✔ Optimización por Colonia de Hormigas (ACO Clásico) completada exitosamente.{RESET}")
        return self.gbest_centroides, self.gbest_asignacion, self.gbest_sse


def graficar_resultados(optimizador, X_datos, y_reales, centroides_aco, labels_aco, labels_kmeans, labels_random, output_path):
    """
    Genera y guarda una gráfica explicativa de 4 paneles con la convergencia, la proyección PCA 2D
    de los clústeres, la matriz de confusión/correspondencia y las métricas comparativas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Agrupamiento (Clustering) con ACO Clásico Discreto (τ_ik y η_ik)\n{nombre_dataset}", fontsize=14, fontweight='bold')

    # Panel 1: Curva de convergencia ACO (Inercia / SSE)
    iteraciones = list(range(1, len(optimizador.historial_gbest_sse) + 1))
    axes[0, 0].plot(iteraciones, optimizador.historial_gbest_sse, 'g-o', linewidth=2.5, label="Mejor Global ACO (gbest SSE)")
    axes[0, 0].plot(iteraciones, optimizador.historial_promedio_sse, 'b--s', linewidth=1.5, alpha=0.7, label="Promedio Colonia (SSE)")
    axes[0, 0].set_title("Convergencia ACO Clásico: Minimización de Inercia (SSE)")
    axes[0, 0].set_xlabel("Iteración ACO")
    axes[0, 0].set_ylabel("Suma de Errores al Cuadrado (SSE)")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend()

    # Panel 2: Visualización de Clústeres (Proyección PCA 2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_datos)
    centroides_pca = pca.transform(centroides_aco)

    colores = ['#e74c3c', '#2ecc71', '#3498db']
    for k in range(N_CLUSTERS):
        mascara = (labels_aco == k)
        axes[0, 1].scatter(X_pca[mascara, 0], X_pca[mascara, 1], c=colores[k], label=f"Clúster ACO {k+1}", alpha=0.7, edgecolors='k', s=50)

    # Dibujar centroides descubiertos por ACO con grandes estrellas doradas
    axes[0, 1].scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='gold', marker='*', s=350, edgecolors='black', linewidth=1.5, label="Centroides ACO (★)", zorder=10)
    axes[0, 1].set_title("Proyección PCA 2D de Clústeres y Centroides ACO")
    axes[0, 1].set_xlabel("Componente Principal 1")
    axes[0, 1].set_ylabel("Componente Principal 2")
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()

    # Panel 3: Matriz de Correspondencia / Confusión entre Clústeres ACO y Clases Reales
    cm = confusion_matrix(y_reales, labels_aco)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[1, 0], cmap='Blues', colorbar=False)
    axes[1, 0].set_title("Matriz de Correspondencia (Clústeres ACO vs Clases Reales)")

    # Panel 4: Comparativa de Coeficientes de Silueta (Silhouette Score)
    sil_aco = silhouette_score(X_datos, labels_aco)
    sil_km = silhouette_score(X_datos, labels_kmeans)
    sil_rnd = silhouette_score(X_datos, labels_random)

    metodos = ['Asignación\nAleatoria', 'K-Means\nEstándar', 'ACO Clásico\n(Discreto τ_ik)']
    valores_silueta = [sil_rnd, sil_km, sil_aco]
    colores_barras = ['#e74c3c', '#f39c12', '#2ecc71']

    bars = axes[1, 1].bar(metodos, valores_silueta, color=colores_barras, width=0.45)
    axes[1, 1].set_title("Comparativa de Coeficiente de Silueta (Silhouette Score)")
    axes[1, 1].set_ylabel("Coeficiente de Silueta [-1, 1]")
    axes[1, 1].set_ylim(-0.2, 0.5)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        axes[1, 1].annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n{MAGENTA}📊 Gráfica explicativa guardada en: {BOLD}{output_path}{RESET}")


# =============================================================================
# 4. FUNCIÓN PRINCIPAL MAIN Y EJECUCIÓN
# =============================================================================
def main():
    # Fijar semilla de NumPy para reproductibilidad estricta de ACO
    np.random.seed(42)

    print(f"{BOLD}{MAGENTA}")
    print("=" * 78)
    print("  AGRUPAMIENTO (CLUSTERING) CON ACO CLÁSICO DISCRETO (τ_ik y η_ik)")
    print("=" * 78)
    print(f"{RESET}")

    print(f"Dataset Seleccionado: {BOLD}{GREEN}{nombre_dataset}{RESET}")
    print(f"Muestras Totales: {BOLD}{n_muestras}{RESET} | Características Químicas: {BOLD}{n_caracteristicas}{RESET} | Grupos (K): {BOLD}{N_CLUSTERS}{RESET}\n")

    # 1. Ejecutar Algoritmo ACO Clásico Discreto
    tiempo_inicio = time.time()
    optimizador = OptimizadorACOClasico(num_hormigas=NUM_HORMIGAS, max_iter=MAX_ITERACIONES)
    centroides_aco, labels_aco, mejor_sse_aco = optimizador.optimizar(X_scaled)
    tiempo_total = time.time() - tiempo_inicio

    sil_aco = silhouette_score(X_scaled, labels_aco)
    ari_aco = adjusted_rand_score(df_y, labels_aco)

    # 2. Exportar registro de evolución de la colonia a CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_evolucion = pd.DataFrame(optimizador.registros_hormigas)
    csv_evolucion_path = os.path.join(script_dir, "evolucion_aco_clustering.csv")
    df_evolucion.to_csv(csv_evolucion_path, index=False)
    print(f"📁 Registro completo de evolución guardado en: {BOLD}{csv_evolucion_path}{RESET}")

    # 3. Métricas de Comparación con K-Means Estándar y Asignación Aleatoria
    print(f"\n{BOLD}{YELLOW}Evaluando Modelos Comparativos (K-Means Estándar y Asignación Aleatoria)...{RESET}")
    kmeans_base = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(X_scaled)
    labels_kmeans = kmeans_base.labels_
    sse_kmeans = kmeans_base.inertia_
    sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
    ari_kmeans = adjusted_rand_score(df_y, labels_kmeans)

    labels_random = np.random.randint(0, N_CLUSTERS, size=n_muestras)
    sil_random = silhouette_score(X_scaled, labels_random)

    print(f"\n{BOLD}===========================================================================")
    print(f"                        RESUMEN COMPARATIVO FINAL DE CLUSTERING")
    print(f"==========================================================================={RESET}")
    print(f" Métrica Evaluativa        │ Asignación Random │ K-Means Estándar │ ACO Clásico Discreto")
    print(f" ──────────────────────────┼───────────────────┼──────────────────┼─────────────────────")
    print(f" Inercia (SSE / Menor mejor)│        N/A        │     {sse_kmeans:9.2f}    │   {BOLD}{GREEN}{mejor_sse_aco:9.2f}{RESET}")
    print(f" Coeficiente de Silueta    │      {sil_random:6.4f}       │      {sil_kmeans:6.4f}      │   {BOLD}{CYAN}{sil_aco:6.4f}{RESET}")
    print(f" Índice Rand Ajustado (ARI) │      {adjusted_rand_score(df_y, labels_random):6.4f}       │      {ari_kmeans:6.4f}      │   {BOLD}{GREEN}{ari_aco:6.4f}{RESET}")
    print(f"===========================================================================\n")
    print(f"⏱ Tiempo total de optimización ACO Clásico: {BOLD}{tiempo_total:.2f} segundos{RESET}\n")

    # 4. Generar gráfica de 4 paneles
    output_image = os.path.join(script_dir, "aco_clustering_results.png")
    graficar_resultados(optimizador, X_scaled, df_y, centroides_aco, labels_aco, labels_kmeans, labels_random, output_image)


if __name__ == "__main__":
    main()
