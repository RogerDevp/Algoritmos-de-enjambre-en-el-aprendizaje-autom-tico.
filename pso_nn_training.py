"""
Entrenamiento de una Red Neuronal (sin backpropagation) usando
Particle Swarm Optimization (PSO).

Curso: Aprendizaje de Máquina - UNAP
Actividad 03 - Parte: Cristian (Entrenamiento de red neuronal
sin backpropagation, usando algoritmos de enjambre)

Dataset: Iris (sklearn), clasificación de 3 clases con 4 características.
Arquitectura: 4 (entrada) -> 8 (oculta) -> 3 (salida)  => 3 capas.

En vez de usar gradiente (backpropagation), el vector completo de
pesos y bias de la red se codifica como la POSICIÓN de una partícula
en un espacio de búsqueda continuo. Un enjambre de partículas explora
ese espacio y converge, mediante PSO, hacia el conjunto de pesos que
minimiza la función de pérdida (entropía cruzada) sobre el set de
entrenamiento.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------
data = load_iris()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 2. ARQUITECTURA DE LA RED (3 capas: entrada, oculta, salida)
# ---------------------------------------------------------------------
N_IN = X.shape[1]      # 4 características
N_HID = 8              # neuronas en la capa oculta
N_OUT = Y.shape[1]     # 3 clases

# Número total de parámetros (pesos + bias) que debe codificar cada partícula
N_PARAMS = (N_IN * N_HID + N_HID) + (N_HID * N_OUT + N_OUT)


def unpack(theta):
    """Convierte el vector plano de una partícula en las matrices
    de pesos y bias de la red (W1, b1, W2, b2)."""
    idx = 0
    W1 = theta[idx: idx + N_IN * N_HID].reshape(N_IN, N_HID)
    idx += N_IN * N_HID
    b1 = theta[idx: idx + N_HID]
    idx += N_HID
    W2 = theta[idx: idx + N_HID * N_OUT].reshape(N_HID, N_OUT)
    idx += N_HID * N_OUT
    b2 = theta[idx: idx + N_OUT]
    return W1, b1, W2, b2


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(theta, X):
    """Propagación hacia adelante (no hay propagación hacia atrás:
    los pesos NO se ajustan con gradientes, solo se evalúan)."""
    W1, b1, W2, b2 = unpack(theta)
    A1 = relu(X @ W1 + b1)
    A2 = softmax(A1 @ W2 + b2)
    return A2


def fitness(theta, X, Y_onehot):
    """Función de aptitud: entropía cruzada (a minimizar).
    Cuanto menor la pérdida, mejor la posición de la partícula."""
    probs = forward(theta, X)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    loss = -np.mean(np.sum(Y_onehot * np.log(probs), axis=1))
    return loss


def accuracy(theta, X, y_true):
    probs = forward(theta, X)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true)


# ---------------------------------------------------------------------
# 3. PSO — CICLO DEL ALGORITMO DE ENJAMBRE
# ---------------------------------------------------------------------
def pso_train(n_particles=40, n_iter=150, w=0.72, c1=1.5, c2=1.5,
              v_max=0.5, seed=42):
    """
    Ciclo completo de PSO aplicado al entrenamiento de la red:

     1) Representación de la partícula: cada partícula es un vector
        theta en R^N_PARAMS con todos los pesos y bias de la red.
     2) Inicialización del enjambre: posiciones y velocidades
        aleatorias para cada partícula.
     3) Función de aptitud: entropía cruzada de la red sobre el
        set de entrenamiento (fitness()).
     4) Comportamiento de la partícula: en cada iteración, la
        partícula ajusta su velocidad combinando inercia (w), su
        mejor posición personal (pbest) y la mejor posición global
        del enjambre (gbest).
     5) Evolución: el enjambre repite el paso 4 iterativamente,
        y pbest/gbest se actualizan cuando se halla una mejor
        solución (menor pérdida).
     6) Finalización: se detiene al llegar a n_iter iteraciones
        (criterio de parada por número máximo de épocas).
    """
    rng = np.random.default_rng(seed)

    # --- (2) Inicialización del enjambre ---
    positions = rng.uniform(-1, 1, size=(n_particles, N_PARAMS))
    velocities = rng.uniform(-0.1, 0.1, size=(n_particles, N_PARAMS))

    pbest_pos = positions.copy()
    pbest_val = np.array([fitness(p, X_train, Y_train) for p in positions])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [gbest_val]

    # --- (4)+(5) Comportamiento de la partícula y evolución ---
    for it in range(n_iter):
        r1 = rng.random((n_particles, N_PARAMS))
        r2 = rng.random((n_particles, N_PARAMS))

        cognitive = c1 * r1 * (pbest_pos - positions)    # atracción a su propio mejor
        social = c2 * r2 * (gbest_pos - positions)        # atracción al mejor global
        velocities = w * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities

        # Evaluar aptitud de cada partícula en la nueva posición
        values = np.array([fitness(p, X_train, Y_train) for p in positions])

        # Actualizar mejor personal
        improved = values < pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = values[improved]

        # Actualizar mejor global
        best_idx = np.argmin(pbest_val)
        if pbest_val[best_idx] < gbest_val:
            gbest_val = pbest_val[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()

        history.append(gbest_val)

        # Muestra el avance en TODAS las iteraciones
        print(f"Iteración {it + 1:3d}/{n_iter} - "
              f"mejor pérdida (gbest): {gbest_val:.4f}")

    # --- (6) Finalización: n_iter alcanzado ---
    return gbest_pos, gbest_val, history


if __name__ == "__main__":
    print(f"Parámetros a optimizar por cada partícula: {N_PARAMS}")
    print("Entrenando la red neuronal con PSO (sin backpropagation)...\n")

    best_theta, best_loss, history = pso_train()

    train_acc = accuracy(best_theta, X_train, y_train)
    test_acc = accuracy(best_theta, X_test, y_test)

    print("\n--- Resultado final ---")
    print(f"Pérdida final (entropía cruzada) en entrenamiento: {best_loss:.4f}")
    print(f"Exactitud en entrenamiento: {train_acc * 100:.2f}%")
    print(f"Exactitud en prueba:        {test_acc * 100:.2f}%")

    # Curva de convergencia (útil para el informe)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteración")
    plt.ylabel("Mejor pérdida global (gbest)")
    plt.title("Convergencia de PSO entrenando la red neuronal")
    plt.tight_layout()
    plt.savefig("convergencia_pso.png", dpi=150)
    print("\nGráfico de convergencia guardado en convergencia_pso.png")
