"""
================================================================================
  AJUSTE DE HIPERPARÁMETROS MEDIANTE PARTICLE SWARM OPTIMIZATION (PSO) EN ML
================================================================================

  ACLARACIÓN CONCEPTUAL IMPORTANTE PARA LA ENTREGA ACADÉMICA:
  ------------------------------------------------------------------------------
  - PSO NO entrena directamente los pesos de la red neuronal.
  - PSO optimiza los HIPERPARÁMETROS del MLPClassifier (capas ocultas, tasa de
    aprendizaje, regularización L2, función de activación y solver).
  - Los pesos y sesgos internos del MLP se entrenan mediante el algoritmo
    optimizador seleccionado (Adam o SGD) vía Backpropagation.
================================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Silenciar advertencias de convergencia de scikit-learn durante la optimización
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
# 1. CARGA Y PREPROCESAMIENTO DE DATOS (BREAST CANCER WISCONSIN)
# =============================================================================
# Carga del dataset real benchmark desde scikit-learn
dataset_raw = load_breast_cancer()

# Definición explícita de las 30 columnas de características (features)
feature_cols = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Construcción de DataFrame para manipulación estructurada de los datos
df_X = pd.DataFrame(dataset_raw.data, columns=feature_cols)
df_y = pd.Series(dataset_raw.target, name="diagnostico")

# Nombres de las clases objetivo
class_names = ["Maligno", "Benigno"]
nombre_dataset = "Wisconsin Diagnostic Breast Cancer (UCI Dataset)"

# División estratificada sin fugas de datos (75% entrenamiento / 25% prueba)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.25, random_state=42, stratify=df_y
)


# =============================================================================
# 2. DEFINICIÓN DE PARÁMETROS Y VARIABLES DEL ALGORITMO PSO
# =============================================================================
# Mapas categóricos de hiperparámetros (solvers simplificados a optimizadores basados en gradiente)
ACTIVACIONES = ["relu", "tanh", "logistic"]  # 3 opciones de activación
SOLVERS = ["adam", "sgd"]                    # 2 opciones de solver

# Espacio de búsqueda de 6 dimensiones:
# Dim 0: Neuronas Capa Oculta 1  => [16.0, 128.0]
# Dim 1: Neuronas Capa Oculta 2  => [0.0, 64.0] (0 = una sola capa oculta)
# Dim 2: log10(lr_init)          => [-4.0, -1.0] --> lr en [0.0001, 0.1]
# Dim 3: log10(alpha)            => [-5.0, -1.0] --> alpha en [0.00001, 0.1]
# Dim 4: Índice de activación     => [0.0, 2.999999] --> np.floor para regiones iguales 
# Dim 5: Índice de solver         => [0.0, 1.999999] --> np.floor para regiones iguales
LIMITES_BUSQUEDA = [
    (16.0, 128.0),
    (0.0, 64.0),
    (-4.0, -1.0),
    (-5.0, -1.0),
    (0.0, 2.999999),
    (0.0, 1.999999)
]

# Configuración del enjambre PSO
NUM_PARTICULAS = 10
MAX_ITERACIONES = 12
W_INICIAL = 0.8  # Inercia inicial
C1 = 1.8         # Coeficiente cognitivo (pbest)
C2 = 1.8         # Coeficiente social (gbest)


# =============================================================================
# 3. ESTRUCTURA, FUNCIONES Y CONTROLADOR DE PSO
# =============================================================================
class Particula:
    """
    Representa una sola partícula dentro del espacio de hiperparámetros.
    """
    def __init__(self, limites):
        num_dimensiones = len(limites)
        self.posicion = np.zeros(num_dimensiones)
        self.velocidad = np.zeros(num_dimensiones)
        
        for d in range(num_dimensiones):
            low, high = limites[d]
            self.posicion[d] = np.random.uniform(low, high)
            rango = high - low
            self.velocidad[d] = np.random.uniform(-rango * 0.1, rango * 0.1)
            
        self.mejor_posicion = self.posicion.copy()
        self.mejor_score = -np.inf
        self.score_actual = -np.inf

    def actualizar_posicion(self, limites):
        """
        Actualiza la posición aplicando la velocidad con control de fronteras (rebote atenuado).
        """
        self.posicion += self.velocidad
        
        for d in range(len(limites)):
            low, high = limites[d]
            if self.posicion[d] < low:
                self.posicion[d] = low
                self.velocidad[d] *= -0.5
            elif self.posicion[d] > high:
                self.posicion[d] = high
                self.velocidad[d] *= -0.5


def decodificar_posicion(posicion):
    """
    Traduce el vector continuo [h1, h2, log_lr, log_alpha, act_idx, solv_idx]
    a un diccionario de hiperparámetros válido para MLPClassifier.
    Usa np.floor para asignar rangos de probabilidad idénticos a variables categóricas.
    """
    h1_val, h2_val, log_lr, log_alpha, act_idx_val, solv_idx_val = posicion
    
    h1 = int(np.clip(np.round(h1_val), 16, 128))
    h2 = int(np.clip(np.round(h2_val), 0, 64))
    
    hidden_layer_sizes = (h1,) if h2 == 0 else (h1, h2)
    
    learning_rate_init = float(10 ** log_lr)
    alpha = float(10 ** log_alpha)
    
    act_idx = int(np.floor(np.clip(act_idx_val, 0, len(ACTIVACIONES) - 0.000001)))
    solv_idx = int(np.floor(np.clip(solv_idx_val, 0, len(SOLVERS) - 0.000001)))
    
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "learning_rate_init": learning_rate_init,
        "alpha": alpha,
        "activation": ACTIVACIONES[act_idx],
        "solver": SOLVERS[solv_idx]
    }


def funcion_objetivo(posicion, X_train_datos, y_train_datos, cv_folds=3):
    """
    Evalúa el fitness (calidad) de una partícula mediante Validación Cruzada de 3 pliegues.
    Utiliza un Pipeline para ajustar el StandardScaler dentro de cada fold y evitar data leakage.
    """
    params = decodificar_posicion(posicion)
    
    try:
        modelo = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            solver=params["solver"],
            learning_rate_init=params["learning_rate_init"],
            alpha=params["alpha"],
            max_iter=150,
            random_state=42
        )
        
        # Pipeline que garantiza el escalamiento exclusivo dentro de cada pliegue de CV
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", modelo)
        ])
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train_datos, y_train_datos, cv=cv, scoring="accuracy", n_jobs=-1)
        return float(np.mean(scores))
    except Exception as e:
        print(f"  {RED}⚠ Error en evaluación de partícula: {e}{RESET}")
        return 0.0


class OptimizadorPSO:
    """
    Controlador principal del enjambre PSO. Ejecuta la búsqueda iterativa
    y muestra el progreso de cada partícula en la consola en tiempo real.
    """
    def __init__(self, num_particulas=NUM_PARTICULAS, max_iter=MAX_ITERACIONES, 
                 w=W_INICIAL, c1=C1, c2=C2):
        self.num_particulas = num_particulas
        self.max_iter = max_iter
        self.w_inicial = w
        self.c1 = c1
        self.c2 = c2
        self.limites = LIMITES_BUSQUEDA
        
        self.enjambre = [Particula(self.limites) for _ in range(num_particulas)]
        self.gbest_posicion = None
        self.gbest_score = -np.inf
        
        # Historiales de convergencia (incluyen la población inicial como iteración 0)
        self.historial_gbest_score = []
        self.historial_promedio_score = []
        self.registros_particulas = []  # Para exportar DataFrame completo de evolución

    def optimizar(self, X_train_datos, y_train_datos):
        """
        Ejecuta el ciclo de optimización PSO imprimiendo los resultados en tiempo real
        y guardando el registro completo de la evolución del enjambre.
        """
        print(f"\n{BOLD}{CYAN}==========================================================================")
        print(f"       OPTIMIZACIÓN EN TIEMPO REAL CON ALGORITMO DE ENJAMBRE (PSO)")
        print(f"=========================================================================={RESET}")
        print(f" Dataset Real          : {BOLD}{CYAN}{nombre_dataset}{RESET}")
        print(f" Columnas de Entrada   : {BOLD}{len(feature_cols)}{RESET} características médicas")
        print(f" Tamaño de Enjambre    : {BOLD}{self.num_particulas}{RESET} partículas")
        print(f" Iteraciones Máximas   : {BOLD}{self.max_iter}{RESET} iteraciones (más Población Inicial)")
        print(f" Parámetros Físicos PSO: w_inicial={self.w_inicial}, c1 (cognitivo)={self.c1}, c2 (social)={self.c2}\n")
        
        print(f"{BOLD}{YELLOW}--- FASE 0: EVALUACIÓN E INICIALIZACIÓN DEL ENJAMBRE (ITERACIÓN 0) ---{RESET}")
        scores_init = []
        
        for i, particula in enumerate(self.enjambre):
            score = funcion_objetivo(particula.posicion, X_train_datos, y_train_datos)
            particula.score_actual = score
            particula.mejor_score = score
            particula.mejor_posicion = particula.posicion.copy()
            scores_init.append(score)
            
            p_dict = decodificar_posicion(particula.posicion)
            es_nuevo_gbest = False
            
            if score > self.gbest_score:
                self.gbest_score = score
                self.gbest_posicion = particula.posicion.copy()
                es_nuevo_gbest = True
            
            # Guardar en el historial completo
            self.registros_particulas.append({
                "iteracion": 0,
                "particula": i + 1,
                "cv_accuracy": score,
                "hidden_layer_sizes": str(p_dict["hidden_layer_sizes"]),
                "activation": p_dict["activation"],
                "solver": p_dict["solver"],
                "learning_rate_init": p_dict["learning_rate_init"],
                "alpha": p_dict["alpha"]
            })
                
            tag_gbest = f" {BOLD}{GREEN}[🔥 NUEVO BEST GLOBAL! Acc = {score*100:.2f}%]{RESET}" if es_nuevo_gbest else ""
            simbolo = "└─" if i == self.num_particulas - 1 else "├─"
            print(f"  {simbolo} Partícula {i+1:02d}/{self.num_particulas:02d} │ CV Acc: {BOLD}{score*100:6.2f}%{RESET} │ Hidden={str(p_dict['hidden_layer_sizes']):10s} │ Act={p_dict['activation']:8s} │ Solver={p_dict['solver']:5s} │ lr={p_dict['learning_rate_init']:.5f}{tag_gbest}")
            sys.stdout.flush()

        # Registrar la Iteración 0 en las curvas de convergencia
        self.historial_gbest_score.append(self.gbest_score)
        self.historial_promedio_score.append(float(np.mean(scores_init)))

        # Bucle principal de iteraciones (1 a MAX_ITERACIONES)
        for iteracion in range(1, self.max_iter + 1):
            w_actual = self.w_inicial * (1.0 - 0.4 * (iteracion / self.max_iter))
            scores_iteracion = []
            
            print(f"\n{BOLD}{MAGENTA}┌─────────────────────────────────────────────────────────────────────────┐{RESET}")
            print(f"{BOLD}{MAGENTA}│ ITERACIÓN {iteracion:02d}/{self.max_iter:02d}  (Factor de Inercia w = {w_actual:.3f})                              │{RESET}")
            print(f"{BOLD}{MAGENTA}└─────────────────────────────────────────────────────────────────────────┘{RESET}")
            
            for i, particula in enumerate(self.enjambre):
                r1 = np.random.rand(len(self.limites))
                r2 = np.random.rand(len(self.limites))
                
                # Fórmula de actualización de velocidad en PSO:
                # v_i = w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
                inercia = w_actual * particula.velocidad
                cognitivo = self.c1 * r1 * (particula.mejor_posicion - particula.posicion)
                social = self.c2 * r2 * (self.gbest_posicion - particula.posicion)
                
                particula.velocidad = inercia + cognitivo + social
                
                # Clamping de velocidad (máximo 25% del rango por dimensión)
                for d in range(len(self.limites)):
                    rango = self.limites[d][1] - self.limites[d][0]
                    v_max = rango * 0.25
                    particula.velocidad[d] = np.clip(particula.velocidad[d], -v_max, v_max)
                
                particula.actualizar_posicion(self.limites)
                
                score = funcion_objetivo(particula.posicion, X_train_datos, y_train_datos)
                particula.score_actual = score
                scores_iteracion.append(score)
                
                p_dict = decodificar_posicion(particula.posicion)
                tag_estado = ""
                
                if score > particula.mejor_score:
                    particula.mejor_score = score
                    particula.mejor_posicion = particula.posicion.copy()
                    tag_estado = f" {CYAN}[★ pbest]{RESET}"
                    
                    if score > self.gbest_score:
                        self.gbest_score = score
                        self.gbest_posicion = particula.posicion.copy()
                        tag_estado = f" {BOLD}{GREEN}[🔥 NUEVO BEST GLOBAL (gbest)! Acc = {score*100:.2f}%]{RESET}"
                
                # Guardar en el historial completo de partículas
                self.registros_particulas.append({
                    "iteracion": iteracion,
                    "particula": i + 1,
                    "cv_accuracy": score,
                    "hidden_layer_sizes": str(p_dict["hidden_layer_sizes"]),
                    "activation": p_dict["activation"],
                    "solver": p_dict["solver"],
                    "learning_rate_init": p_dict["learning_rate_init"],
                    "alpha": p_dict["alpha"]
                })
                
                simbolo = "└─" if i == self.num_particulas - 1 else "├─"
                print(f"  {simbolo} Partícula {i+1:02d}/{self.num_particulas:02d} │ CV Acc: {score*100:6.2f}% │ Hidden={str(p_dict['hidden_layer_sizes']):10s} │ Act={p_dict['activation']:8s} │ Solver={p_dict['solver']:5s} │ lr={p_dict['learning_rate_init']:.5f}{tag_estado}")
                sys.stdout.flush()

            promedio_iter = float(np.mean(scores_iteracion))
            std_iter = float(np.std(scores_iteracion))
            self.historial_gbest_score.append(self.gbest_score)
            self.historial_promedio_score.append(promedio_iter)
            
            pbest_dict = decodificar_posicion(self.gbest_posicion)
            print(f"\n  {BOLD}{GREEN}▶ RESUMEN ITERACIÓN {iteracion:02d}/{self.max_iter:02d}:{RESET}")
            print(f"    • Mejor CV Accuracy Global (gbest) : {BOLD}{GREEN}{self.gbest_score*100:.2f}%{RESET}")
            print(f"    • Promedio del Enjambre (Fitness)  : {promedio_iter*100:.2f}% (± {std_iter*100:.2f}%)")
            print(f"    • Configuración gbest actual       : Hidden={pbest_dict['hidden_layer_sizes']}, Act='{pbest_dict['activation']}', Solver='{pbest_dict['solver']}', lr={pbest_dict['learning_rate_init']:.5f}, α={pbest_dict['alpha']:.5f}")

        print(f"\n{BOLD}{GREEN}✔ Optimización PSO completada exitosamente.{RESET}")
        return decodificar_posicion(self.gbest_posicion), self.gbest_score


def graficar_resultados(optimizador, acc_base, acc_pso, y_test_datos, y_pred_pso, class_labels, output_path):
    """
    Genera y guarda una gráfica de 3 paneles comparando baseline vs PSO en el dataset real.
    Muestra el eje de iteraciones arrancando desde la Iteración 0 (población inicial).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"Hyperparameter Tuning con PSO sobre Dataset Real:\n{nombre_dataset}", fontsize=13, fontweight='bold')

    # Panel 1: Curva de convergencia PSO (Iteración 0 a MaxIter)
    iteraciones = list(range(0, len(optimizador.historial_gbest_score)))
    axes[0].plot(iteraciones, [s * 100 for s in optimizador.historial_gbest_score], 'g-o', linewidth=2.5, label="Mejor Global (gbest)")
    axes[0].plot(iteraciones, [s * 100 for s in optimizador.historial_promedio_score], 'b--s', linewidth=1.5, alpha=0.7, label="Promedio del Enjambre")
    axes[0].set_title("Convergencia del Enjambre (CV Accuracy %)")
    axes[0].set_xlabel("Iteración PSO (0 = Población Inicial)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Panel 2: Comparativa Baseline vs PSO
    modelos = ['MLP Baseline Estándar\n(Sin Optimizar)', 'MLP Optimizado\n(Enjambre PSO)']
    accuracies = [acc_base * 100, acc_pso * 100]
    colores = ['#3498db', '#2ecc71']
    
    bars = axes[1].bar(modelos, accuracies, color=colores, width=0.45)
    axes[1].set_title("Comparativa en Conjunto de Test (%)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{height:.2f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4),  
                         textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Panel 3: Matriz de Confusión
    cm = confusion_matrix(y_test_datos, y_pred_pso)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=axes[2], cmap='Greens', colorbar=False)
    axes[2].set_title("Matriz de Confusión (Modelo PSO)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n{MAGENTA}📊 Gráfica explicativa guardada en: {BOLD}{output_path}{RESET}")


# =============================================================================
# 4. FUNCIÓN PRINCIPAL MAIN Y EJECUCIÓN
# =============================================================================
def main():
    # Fijar semilla de NumPy para la reproductibilidad estricta de PSO
    np.random.seed(42)

    print(f"{BOLD}{MAGENTA}")
    print("=" * 78)
    print("   AJUSTE DE HIPERPARÁMETROS CON ALGORITMOS DE ENJAMBRE (PSO) EN DATASET REAL")
    print("=" * 78)
    print(f"{RESET}")
    
    print(f"Dataset Seleccionado: {BOLD}{GREEN}{nombre_dataset}{RESET}")
    print(f"Muestras de Entrenamiento: {BOLD}{X_train_raw.shape[0]}{RESET} | Muestras de Test: {BOLD}{X_test_raw.shape[0]}{RESET} | Características: {BOLD}{X_train_raw.shape[1]}{RESET} | Clases: {BOLD}{len(class_names)}{RESET}\n")

    # 1. Modelo Baseline Estándar de Scikit-Learn (Comparativa justa)
    print(f"{BOLD}{YELLOW}1. Evaluando Modelo Baseline Estándar (MLPClassifier por Defecto)...{RESET}")
    pipeline_base = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=500, random_state=42))
    ])
    pipeline_base.fit(X_train_raw, y_train)
    y_pred_base = pipeline_base.predict(X_test_raw)
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"   • Configuración Baseline Estándar: hidden_layers=(100,), activation='relu', solver='adam', lr=0.001")
    print(f"   • Accuracy en Test del Baseline: {BOLD}{acc_base * 100:.2f}%{RESET}\n")

    # 2. Optimización con PSO mostrando resultados en tiempo real
    tiempo_inicio = time.time()
    optimizador = OptimizadorPSO(num_particulas=NUM_PARTICULAS, max_iter=MAX_ITERACIONES)
    mejores_params, mejor_cv_score = optimizador.optimizar(X_train_raw, y_train)
    tiempo_total = time.time() - tiempo_inicio

    # 3. Exportar historial completo de evolución a archivo CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_evolucion = pd.DataFrame(optimizador.registros_particulas)
    csv_evolucion_path = os.path.join(script_dir, "evolucion_pso.csv")
    df_evolucion.to_csv(csv_evolucion_path, index=False)
    print(f"📁 Registro completo de evolución guardado en: {BOLD}{csv_evolucion_path}{RESET}")

    # 4. Evaluación del modelo optimizado por PSO
    print(f"\n{BOLD}{GREEN}2. Entrenando Modelo Final con los Mejores Hiperparámetros del Enjambre...{RESET}")
    print(f"   • Tiempo transcurrido en PSO: {tiempo_total:.2f} segundos")
    print(f"   • Mejores Hiperparámetros Encontrados:")
    print(f"     - Arquitectura Capas Ocultas : {BOLD}{mejores_params['hidden_layer_sizes']}{RESET}")
    print(f"     - Función de Activación      : {BOLD}{mejores_params['activation']}{RESET}")
    print(f"     - Algoritmo Optimizador      : {BOLD}{mejores_params['solver']}{RESET}")
    print(f"     - Tasa de Aprendizaje (lr)   : {BOLD}{mejores_params['learning_rate_init']:.6f}{RESET}")
    print(f"     - Regularización L2 (alpha)  : {BOLD}{mejores_params['alpha']:.6f}{RESET}")

    modelo_pso = MLPClassifier(
        hidden_layer_sizes=mejores_params["hidden_layer_sizes"],
        activation=mejores_params["activation"],
        solver=mejores_params["solver"],
        learning_rate_init=mejores_params["learning_rate_init"],
        alpha=mejores_params["alpha"],
        max_iter=500,
        random_state=42
    )
    
    pipeline_pso = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", modelo_pso)
    ])
    pipeline_pso.fit(X_train_raw, y_train)
    y_pred_pso = pipeline_pso.predict(X_test_raw)
    acc_pso = accuracy_score(y_test, y_pred_pso)

    mejora_abs = (acc_pso - acc_base) * 100

    print(f"\n{BOLD}===========================================================================")
    print(f"                        RESUMEN COMPARATIVO FINAL")
    print(f"==========================================================================={RESET}")
    print(f" Accuracy Baseline Estándar        : {BOLD}{acc_base * 100:6.2f}%{RESET}")
    print(f" Accuracy Optimizado con PSO       : {BOLD}{GREEN}{acc_pso * 100:6.2f}%{RESET}")
    print(f" Diferencia Absoluta de Rendimiento: {BOLD}{CYAN}{'+' if mejora_abs >= 0 else ''}{mejora_abs:6.2f}%{RESET}")
    print(f"===========================================================================\n")
    
    print(f"{BOLD}Reporte Detallado de Clasificación (Modelo Optimizado por PSO):{RESET}")
    print(classification_report(y_test, y_pred_pso, target_names=class_names))

    # 5. Generar gráfica explicativa
    output_image = os.path.join(script_dir, "pso_tuning_results.png")
    graficar_resultados(optimizador, acc_base, acc_pso, y_test, y_pred_pso, class_names, output_image)


if __name__ == "__main__":
    main()
