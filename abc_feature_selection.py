# ============================================================
# SELECCIÓN DE CARACTERÍSTICAS CON ARTIFICIAL BEE COLONY
# MLP CLASSIFIER
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import kagglehub


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

HIDDEN_LAYERS = (16, 8, 4)
ACTIVATION = "relu"
SOLVER = "adam"
LEARNING_RATE_INIT = 0.001
MAX_ITER = 1000


# ============================================================
# PARÁMETROS DEL ALGORITMO ABC
# ============================================================

NUM_BEES = 10
MAX_ITERATIONS = 20
LIMIT = 5

LAMBDA = 0.10

# Probabilidad de realizar una mutación adicional
MUTATION_PROBABILITY = 0.05


# ============================================================
# CARPETA DE RESULTADOS
# ============================================================

RESULTS_DIR = "abc_feature_selection"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# DESCARGA DEL DATASET
# ============================================================

path = kagglehub.dataset_download(
    "lainguyn123/student-performance-factors"
)

print("Ruta del dataset:", path)


# ============================================================
# CARGA DEL CSV
# ============================================================

archivos_csv = [
    archivo
    for archivo in os.listdir(path)
    if archivo.lower().endswith(".csv")
]

if not archivos_csv:
    raise FileNotFoundError(
        "No se encontró ningún archivo CSV en el dataset."
    )

ruta_csv = os.path.join(
    path,
    archivos_csv[0]
)

df = pd.read_csv(
    ruta_csv
)

print("\nDimensiones originales:", df.shape)


# ============================================================
# LIMPIEZA DE DATOS
# ============================================================

df = df.copy()

columnas_categoricas = df.select_dtypes(
    include=["object", "category"]
).columns

for columna in columnas_categoricas:
    df[columna] = df[columna].fillna(
        df[columna].mode()[0]
    )

columnas_numericas = df.select_dtypes(
    include=[np.number]
).columns

for columna in columnas_numericas:
    df[columna] = df[columna].fillna(
        df[columna].median()
    )


# ============================================================
# CREACIÓN DE LA VARIABLE OBJETIVO
# ============================================================

if "Exam_Score" not in df.columns:
    raise ValueError(
        "No se encontró la columna 'Exam_Score'."
    )

df["performance_level"] = pd.qcut(
    df["Exam_Score"],
    q=3,
    labels=[0, 1, 2]
)

y = df["performance_level"].astype(int)


# ============================================================
# VARIABLES PREDICTORAS
# ============================================================

X = df.drop(
    columns=[
        "Exam_Score",
        "performance_level"
    ]
)


# ============================================================
# CODIFICACIÓN
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

X = X.astype(float)

n_features = X.shape[1]

nombres_caracteristicas = X.columns.tolist()

print(
    "\nNúmero total de características:",
    n_features
)


# ============================================================
# DIVISIÓN TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# DIVISIÓN TRAIN / VALIDACIÓN PARA ABC
# ============================================================

X_abc_train, X_validacion, y_abc_train, y_validacion = (
    train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE
    )
)


print(
    "\nABC entrenamiento:",
    X_abc_train.shape
)

print(
    "ABC validación:",
    X_validacion.shape
)

print(
    "Prueba final:",
    X_test.shape
)


# ============================================================
# FUNCIÓN PARA ENTRENAR Y EVALUAR UNA SOLUCIÓN
# ============================================================

def evaluar_solucion(
    solucion,
    X_entrenamiento,
    y_entrenamiento,
    X_validacion,
    y_validacion
):

    indices = np.where(
        solucion == 1
    )[0]

    numero_seleccionadas = len(indices)

    # No permitir soluciones sin características
    if numero_seleccionadas == 0:

        return {
            "accuracy": 0.0,
            "fitness": -1.0,
            "n_caracteristicas": 0
        }

    X_train_seleccionado = (
        X_entrenamiento.iloc[:, indices]
    )

    X_validacion_seleccionada = (
        X_validacion.iloc[:, indices]
    )

    # --------------------------------------------------------
    # ESCALAMIENTO
    # --------------------------------------------------------

    escalador = StandardScaler()

    X_train_escalado = (
        escalador.fit_transform(
            X_train_seleccionado
        )
    )

    X_validacion_escalado = (
        escalador.transform(
            X_validacion_seleccionada
        )
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    modelo = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation=ACTIVATION,
        solver=SOLVER,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE
    )

    modelo.fit(
        X_train_escalado,
        y_entrenamiento
    )

    predicciones = modelo.predict(
        X_validacion_escalado
    )

    accuracy = accuracy_score(
        y_validacion,
        predicciones
    )

    # --------------------------------------------------------
    # FUNCIÓN FITNESS
    # --------------------------------------------------------

    penalizacion = (
        LAMBDA *
        numero_seleccionadas /
        n_features
    )

    fitness = (
        accuracy -
        penalizacion
    )

    return {
        "accuracy": accuracy,
        "fitness": fitness,
        "n_caracteristicas": numero_seleccionadas
    }


# ============================================================
# GENERACIÓN DE SOLUCIÓN ALEATORIA
# ============================================================

def generar_solucion():

    solucion = np.random.randint(
        0,
        2,
        size=n_features
    )

    # Garantizar al menos una característica
    if np.sum(solucion) == 0:

        indice = np.random.randint(
            0,
            n_features
        )

        solucion[indice] = 1

    return solucion


# ============================================================
# CREAR NUEVA SOLUCIÓN
# ============================================================

def generar_candidata(
    solucion_actual,
    otra_solucion
):

    candidata = solucion_actual.copy()

    # Seleccionar una posición diferente
    posiciones_diferentes = np.where(
        solucion_actual != otra_solucion
    )[0]

    if len(posiciones_diferentes) > 0:

        posicion = np.random.choice(
            posiciones_diferentes
        )

        candidata[posicion] = (
            otra_solucion[posicion]
        )

    # Mutación adicional
    if np.random.rand() < MUTATION_PROBABILITY:

        posicion = np.random.randint(
            0,
            n_features
        )

        candidata[posicion] = (
            1 - candidata[posicion]
        )

    # Evitar solución vacía
    if np.sum(candidata) == 0:

        posicion = np.random.randint(
            0,
            n_features
        )

        candidata[posicion] = 1

    return candidata


# ============================================================
# INICIALIZACIÓN
# ============================================================

np.random.seed(
    RANDOM_STATE
)

abejas = [
    generar_solucion()
    for _ in range(NUM_BEES)
]

fitness = np.zeros(
    NUM_BEES
)

accuracies = np.zeros(
    NUM_BEES
)

numero_caracteristicas = np.zeros(
    NUM_BEES,
    dtype=int
)

contadores_sin_mejora = np.zeros(
    NUM_BEES,
    dtype=int
)


# ============================================================
# HISTORIALES
# ============================================================

historial_evaluaciones = []
historial_mejor = []
historial_eventos = []


# ============================================================
# EVALUACIÓN INICIAL
# ============================================================

print("\n" + "=" * 70)
print("INICIALIZACIÓN DE LA COLONIA")
print("=" * 70)

for i in range(NUM_BEES):

    resultado = evaluar_solucion(
        abejas[i],
        X_abc_train,
        y_abc_train,
        X_validacion,
        y_validacion
    )

    accuracies[i] = resultado["accuracy"]
    fitness[i] = resultado["fitness"]
    numero_caracteristicas[i] = (
        resultado["n_caracteristicas"]
    )

    historial_evaluaciones.append({
        "fase": "inicializacion",
        "iteracion": 0,
        "abeja": i + 1,
        "accuracy": accuracies[i],
        "fitness": fitness[i],
        "n_caracteristicas": numero_caracteristicas[i]
    })


# ============================================================
# MEJOR SOLUCIÓN INICIAL
# ============================================================

indice_mejor = np.argmax(
    fitness
)

mejor_solucion = abejas[
    indice_mejor
].copy()

mejor_fitness = fitness[
    indice_mejor
]

mejor_accuracy = accuracies[
    indice_mejor
]

mejor_n_caracteristicas = numero_caracteristicas[
    indice_mejor
]


historial_mejor.append({
    "iteracion": 0,
    "fitness": mejor_fitness,
    "accuracy": mejor_accuracy,
    "n_caracteristicas": mejor_n_caracteristicas
})


print(
    f"Mejor fitness inicial: "
    f"{mejor_fitness:.4f}"
)

print(
    f"Accuracy inicial: "
    f"{mejor_accuracy:.4f}"
)

print(
    f"Características iniciales: "
    f"{mejor_n_caracteristicas}"
)


# ============================================================
# ALGORITMO ARTIFICIAL BEE COLONY
# ============================================================

for iteracion in range(
    1,
    MAX_ITERATIONS + 1
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"ITERACIÓN {iteracion}/{MAX_ITERATIONS}"
    )

    print(
        f"{'=' * 70}"
    )

    # --------------------------------------------------------
    # FASE DE ABEJAS EMPLEADAS
    # --------------------------------------------------------

    for i in range(NUM_BEES):

        otro_indice = np.random.choice(
            [
                j
                for j in range(NUM_BEES)
                if j != i
            ]
        )

        candidata = generar_candidata(
            abejas[i],
            abejas[otro_indice]
        )

        resultado = evaluar_solucion(
            candidata,
            X_abc_train,
            y_abc_train,
            X_validacion,
            y_validacion
        )

        historial_evaluaciones.append({
            "fase": "empleadas",
            "iteracion": iteracion,
            "abeja": i + 1,
            "accuracy": resultado["accuracy"],
            "fitness": resultado["fitness"],
            "n_caracteristicas": resultado[
                "n_caracteristicas"
            ]
        })

        # ----------------------------------------------------
        # SELECCIÓN GREEDY
        # ----------------------------------------------------

        if resultado["fitness"] > fitness[i]:

            abejas[i] = candidata

            fitness[i] = (
                resultado["fitness"]
            )

            accuracies[i] = (
                resultado["accuracy"]
            )

            numero_caracteristicas[i] = (
                resultado["n_caracteristicas"]
            )

            contadores_sin_mejora[i] = 0

            historial_eventos.append({
                "iteracion": iteracion,
                "fase": "empleadas",
                "abeja": i + 1,
                "evento": "mejora"
            })

        else:

            contadores_sin_mejora[i] += 1

            historial_eventos.append({
                "iteracion": iteracion,
                "fase": "empleadas",
                "abeja": i + 1,
                "evento": "sin_mejora"
            })


    # --------------------------------------------------------
    # FASE DE ABEJAS OBSERVADORAS
    # --------------------------------------------------------

    fitness_minimo = np.min(
        fitness
    )

    valores_probabilidad = (
        fitness - fitness_minimo + 1e-10
    )

    probabilidades = (
        valores_probabilidad /
        np.sum(valores_probabilidad)
    )

    for _ in range(NUM_BEES):

        i = np.random.choice(
            np.arange(NUM_BEES),
            p=probabilidades
        )

        otro_indice = np.random.choice(
            [
                j
                for j in range(NUM_BEES)
                if j != i
            ]
        )

        candidata = generar_candidata(
            abejas[i],
            abejas[otro_indice]
        )

        resultado = evaluar_solucion(
            candidata,
            X_abc_train,
            y_abc_train,
            X_validacion,
            y_validacion
        )

        historial_evaluaciones.append({
            "fase": "observadoras",
            "iteracion": iteracion,
            "abeja": i + 1,
            "accuracy": resultado["accuracy"],
            "fitness": resultado["fitness"],
            "n_caracteristicas": resultado[
                "n_caracteristicas"
            ]
        })

        if resultado["fitness"] > fitness[i]:

            abejas[i] = candidata

            fitness[i] = (
                resultado["fitness"]
            )

            accuracies[i] = (
                resultado["accuracy"]
            )

            numero_caracteristicas[i] = (
                resultado["n_caracteristicas"]
            )

            contadores_sin_mejora[i] = 0

            historial_eventos.append({
                "iteracion": iteracion,
                "fase": "observadoras",
                "abeja": i + 1,
                "evento": "mejora"
            })

        else:

            contadores_sin_mejora[i] += 1

            historial_eventos.append({
                "iteracion": iteracion,
                "fase": "observadoras",
                "abeja": i + 1,
                "evento": "sin_mejora"
            })


    # --------------------------------------------------------
    # FASE DE ABEJAS EXPLORADORAS / SCOUT
    # --------------------------------------------------------

    for i in range(NUM_BEES):

        if contadores_sin_mejora[i] >= LIMIT:

            abejas[i] = generar_solucion()

            resultado = evaluar_solucion(
                abejas[i],
                X_abc_train,
                y_abc_train,
                X_validacion,
                y_validacion
            )

            fitness[i] = (
                resultado["fitness"]
            )

            accuracies[i] = (
                resultado["accuracy"]
            )

            numero_caracteristicas[i] = (
                resultado["n_caracteristicas"]
            )

            contadores_sin_mejora[i] = 0

            historial_evaluaciones.append({
                "fase": "exploradoras",
                "iteracion": iteracion,
                "abeja": i + 1,
                "accuracy": resultado["accuracy"],
                "fitness": resultado["fitness"],
                "n_caracteristicas": resultado[
                    "n_caracteristicas"
                ]
            })

            historial_eventos.append({
                "iteracion": iteracion,
                "fase": "exploradoras",
                "abeja": i + 1,
                "evento": "reinicio_scout"
            })


    # --------------------------------------------------------
    # ACTUALIZAR MEJOR SOLUCIÓN GLOBAL
    # --------------------------------------------------------

    indice_mejor_iteracion = np.argmax(
        fitness
    )

    if (
        fitness[indice_mejor_iteracion]
        > mejor_fitness
    ):

        mejor_solucion = abejas[
            indice_mejor_iteracion
        ].copy()

        mejor_fitness = fitness[
            indice_mejor_iteracion
        ]

        mejor_accuracy = accuracies[
            indice_mejor_iteracion
        ]

        mejor_n_caracteristicas = (
            numero_caracteristicas[
                indice_mejor_iteracion
            ]
        )

        evento_mejor = "nuevo_mejor"

    else:

        evento_mejor = "sin_nuevo_mejor"


    historial_mejor.append({
        "iteracion": iteracion,
        "fitness": mejor_fitness,
        "accuracy": mejor_accuracy,
        "n_caracteristicas": (
            mejor_n_caracteristicas
        )
    })

    print(
        f"Mejor fitness: "
        f"{mejor_fitness:.4f}"
    )

    print(
        f"Mejor accuracy: "
        f"{mejor_accuracy:.4f}"
    )

    print(
        f"Características: "
        f"{mejor_n_caracteristicas}"
    )

    print(
        f"Evento: {evento_mejor}"
    )


# ============================================================
# CARACTERÍSTICAS SELECCIONADAS
# ============================================================

indices_seleccionados = np.where(
    mejor_solucion == 1
)[0]

caracteristicas_seleccionadas = [
    nombres_caracteristicas[i]
    for i in indices_seleccionados
]

caracteristicas_no_seleccionadas = [
    nombres_caracteristicas[i]
    for i in range(n_features)
    if mejor_solucion[i] == 0
]


print("\n" + "=" * 70)
print("RESULTADO FINAL DE ABC")
print("=" * 70)

print(
    f"\nCaracterísticas originales: "
    f"{n_features}"
)

print(
    f"Características seleccionadas: "
    f"{len(caracteristicas_seleccionadas)}"
)

print(
    f"Reducción: "
    f"{100 * (1 - len(caracteristicas_seleccionadas) / n_features):.2f}%"
)

print(
    f"Accuracy validación: "
    f"{mejor_accuracy:.4f}"
)

print(
    f"Fitness: "
    f"{mejor_fitness:.4f}"
)

print("\nCaracterísticas seleccionadas:")

for caracteristica in caracteristicas_seleccionadas:
    print(
        " -",
        caracteristica
    )


# ============================================================
# ENTRENAMIENTO FINAL CON LAS CARACTERÍSTICAS SELECCIONADAS
# ============================================================

X_train_seleccionado = (
    X_train.iloc[:, indices_seleccionados]
)

X_test_seleccionado = (
    X_test.iloc[:, indices_seleccionados]
)


# ------------------------------------------------------------
# ESCALAMIENTO FINAL
# ------------------------------------------------------------

escalador_final = StandardScaler()

X_train_final = (
    escalador_final.fit_transform(
        X_train_seleccionado
    )
)

X_test_final = (
    escalador_final.transform(
        X_test_seleccionado
    )
)


# ============================================================
# MODELO FINAL
# ============================================================

modelo_final = MLPClassifier(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation=ACTIVATION,
    solver=SOLVER,
    learning_rate_init=LEARNING_RATE_INIT,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE
)


# ============================================================
# ENTRENAMIENTO FINAL
# ============================================================

modelo_final.fit(
    X_train_final,
    y_train
)


# ============================================================
# EVALUACIÓN FINAL SOBRE TEST
# ============================================================

y_pred_test = modelo_final.predict(
    X_test_final
)

accuracy_test = accuracy_score(
    y_test,
    y_pred_test
)

matriz_test = confusion_matrix(
    y_test,
    y_pred_test
)


print("\n" + "=" * 70)
print("EVALUACIÓN FINAL - MLP + ABC")
print("=" * 70)

print(
    f"\nAccuracy test: "
    f"{accuracy_test:.4f}"
)

print("\nMatriz de confusión:")

print(
    matriz_test
)


# ============================================================
# CSV: HISTORIAL DE EVALUACIONES
# ============================================================

df_historial_evaluaciones = pd.DataFrame(
    historial_evaluaciones
)

df_historial_evaluaciones.to_csv(
    os.path.join(
        RESULTS_DIR,
        "abc_historial_evaluaciones.csv"
    ),
    index=False
)


# ============================================================
# CSV: HISTORIAL DEL MEJOR
# ============================================================

df_historial_mejor = pd.DataFrame(
    historial_mejor
)

df_historial_mejor.to_csv(
    os.path.join(
        RESULTS_DIR,
        "abc_historial_mejor.csv"
    ),
    index=False
)


# ============================================================
# CSV: EVENTOS
# ============================================================

df_historial_eventos = pd.DataFrame(
    historial_eventos
)

df_historial_eventos.to_csv(
    os.path.join(
        RESULTS_DIR,
        "abc_historial_eventos.csv"
    ),
    index=False
)


# ============================================================
# FRECUENCIA DE SELECCIÓN
# ============================================================

frecuencia_seleccion = pd.DataFrame({
    "caracteristica": nombres_caracteristicas,
    "seleccionada": mejor_solucion
})

frecuencia_seleccion[
    "seleccionada"
] = frecuencia_seleccion[
    "seleccionada"
].astype(int)


frecuencia_seleccion.to_csv(
    os.path.join(
        RESULTS_DIR,
        "abc_frecuencia_caracteristicas.csv"
    ),
    index=False
)


# ============================================================
# CSV: RESULTADO FINAL
# ============================================================

resultado_final = pd.DataFrame({
    "modelo": [
        "MLP Base",
        "MLP + ABC"
    ],
    "accuracy": [
        np.nan,
        accuracy_test
    ],
    "fitness": [
        np.nan,
        mejor_fitness
    ],
    "caracteristicas": [
        n_features,
        len(caracteristicas_seleccionadas)
    ]
})

resultado_final.to_csv(
    os.path.join(
        RESULTS_DIR,
        "abc_resultado_final.csv"
    ),
    index=False
)


# ============================================================
# GRÁFICA 1: CONVERGENCIA ABC
# ============================================================

plt.figure()

plt.plot(
    df_historial_mejor["iteracion"],
    df_historial_mejor["fitness"],
    marker="o"
)

plt.xlabel(
    "Iteración"
)

plt.ylabel(
    "Mejor fitness"
)

plt.title(
    "Convergencia del algoritmo ABC"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "01_convergencia_abc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICA 2: ACCURACY VS CARACTERÍSTICAS
# ============================================================

plt.figure()

plt.scatter(
    df_historial_evaluaciones[
        "n_caracteristicas"
    ],
    df_historial_evaluaciones[
        "accuracy"
    ],
    alpha=0.5
)

plt.xlabel(
    "Número de características"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Accuracy vs. número de características"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "02_accuracy_vs_caracteristicas.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICA 3: FITNESS VS CARACTERÍSTICAS
# ============================================================

plt.figure()

plt.scatter(
    df_historial_evaluaciones[
        "n_caracteristicas"
    ],
    df_historial_evaluaciones[
        "fitness"
    ],
    alpha=0.5
)

plt.xlabel(
    "Número de características"
)

plt.ylabel(
    "Fitness"
)

plt.title(
    "Fitness vs. número de características"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "03_fitness_vs_caracteristicas.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICA 4: EVOLUCIÓN DEL NÚMERO DE CARACTERÍSTICAS
# ============================================================

plt.figure()

plt.plot(
    df_historial_mejor["iteracion"],
    df_historial_mejor[
        "n_caracteristicas"
    ],
    marker="o"
)

plt.xlabel(
    "Iteración"
)

plt.ylabel(
    "Número de características"
)

plt.title(
    "Evolución de características seleccionadas"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "04_evolucion_numero_caracteristicas.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICA 5: FRECUENCIA DE SELECCIÓN
# ============================================================

df_frecuencia = (
    frecuencia_seleccion
    .sort_values(
        "seleccionada",
        ascending=False
    )
)

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    df_frecuencia[
        "caracteristica"
    ],
    df_frecuencia[
        "seleccionada"
    ]
)

plt.xlabel(
    "Características"
)

plt.ylabel(
    "Seleccionada"
)

plt.title(
    "Características seleccionadas por ABC"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "05_frecuencia_seleccion.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICA 6: MATRIZ DE CONFUSIÓN
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz_test,
    display_labels=[
        "Bajo",
        "Medio",
        "Alto"
    ]
)

disp.plot()

plt.title(
    "Matriz de confusión - MLP + ABC"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "06_matriz_confusion_abc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)

print(
    f"\nResultados: {RESULTS_DIR}/"
)

for archivo in sorted(
    os.listdir(RESULTS_DIR)
):

    print(
        " -",
        archivo
    )

print("\nProceso ABC finalizado correctamente.")