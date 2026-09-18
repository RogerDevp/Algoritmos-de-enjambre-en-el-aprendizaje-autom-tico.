# Algoritmos-de-enjambre-en-el-aprendizaje-autom-tico.
# Actividad 03 – Algoritmos de Enjambre

## Parte 1: Selección de Características con algoritmo ABC 

Se implementa el **Algoritmo de Colonia de Abejas Artificiales (ABC)** aplicado a la selección de características sobre el dataset *Student Performance Factors*. El objetivo principal es optimizar la precisión de clasificación de un modelo `MLPClassifier` al mismo tiempo que se reduce la dimensionalidad de las variables de entrada.

### Flujo de Trabajo y Arquitectura del Algoritmo

### 1. Representación del Problema
* **Vector de Solución:** Cada solución se representa mediante una cadena binaria de 19 posiciones (correspondiente a las 19 variables predictoras).
  * `1`: La característica es seleccionada.
  * `0`: La característica es descartada.
* **Función Clave:** `generar_solucion()` crea cadenas aleatorias asegurando que al menos una característica sea seleccionada.

### 2. Función de Fitness (Evaluación)
Las soluciones se evalúan entrenando un `MLPClassifier` únicamente con las características seleccionadas (`1`s). La función de **Fitness** penaliza la cantidad de variables seleccionadas para promover modelos parsimoniosos:

$$\text{Fitness} = \text{Accuracy} - \left(\lambda \times \frac{\text{Características Seleccionadas}}{\text{Total Características}}\right)$$

* **Función Clave:** `evaluar_solucion()` calcula el *accuracy*, el número de variables y el *fitness* final.

### Fases del Algoritmo ABC

1. **Abejas Empleadas (Obreras):**
   * Cada abeja explora la vecindad comparando su solución actual con otra aleatoria mediante la función `generar_candidata()`.
   * Modifica posiciones donde difieren ambas cadenas.
   * Acepta la nueva candidata **únicamente si su Fitness es superior** a la solución actual.

2. **Abejas Observadoras:**
   * Evalúan la calidad de las soluciones existentes y seleccionan cuál explorar utilizando una distribución de probabilidad basada en el *Fitness* (`np.random.choice`).
   * Generan candidatas locales y aplican selección codiciosa (greedy selection).

3. **Abejas Exploradoras (Scouts):**
   * Se lleva un control de intentos fallidos sin mejora (`contadores_sin_mejora`).
   * Si una solución no presenta mejoras tras superar el umbral fijado (`LIMIT = 5`), la solución se abandona.
   * La abeja scout re-inicializa una nueva solución totalmente aleatoria mediante `generar_solucion()` para evitar máximos locales.

Parámetros Principales

| Parámetro        | Valor  | Descripción                                                          |
| :--------------- | :----- | :------------------------------------------------------------------- |
| `NUM_BEES`       | `10`   | Número de abejas en la colonia.                                      |
| `MAX_ITERATIONS` | `20`   | Iteraciones máximas de búsqueda.                                     |
| `LIMIT`          | `5`    | Límite de intentos sin mejora antes de activar la abeja exploradora. |
| `\lambda`        | `0.10` | Factor de penalización por número de características utilizadas.     |

## Tecnologías y Librerías Utilizadas

* **Lenguaje:** Python
* **Librerías Principales:** `numpy`, `scikit-learn` (`MLPClassifier`)


---



## Parte: Entrenamiento de Red Neuronal sin Backpropagation y Clustering

### 1. Entrenamiento de una Red Neuronal mediante PSO

En esta parte se implementó Particle Swarm Optimization (PSO) para entrenar una red neuronal sin utilizar backpropagation ni cálculo de gradientes.

Se utilizó inicialmente el dataset Iris, compuesto por 150 muestras, 4 características y 3 clases.

La red neuronal utilizada está formada por:

- Capa de entrada: 4 neuronas.
- Capa oculta: 8 neuronas con función de activación ReLU.
- Capa de salida: 3 neuronas con Softmax.

En total, la red contiene 67 parámetros entre pesos y bias.

Cada partícula del enjambre representa una configuración completa de los pesos y bias de la red neuronal. La función de aptitud utilizada es la entropía cruzada.

Durante el proceso, PSO actualiza las partículas utilizando su mejor posición personal (pbest) y la mejor posición global encontrada por el enjambre (gbest).

El proceso se repite durante las iteraciones hasta obtener la mejor configuración encontrada.

#### Resultados

En el experimento con Iris se obtuvo:

- Exactitud en entrenamiento: 100 %
- Exactitud en prueba: 94.74 %
- Aciertos en prueba: 36 de 38 muestras.

Esto demuestra que PSO puede utilizarse para encontrar los pesos y bias de una red neuronal sin utilizar backpropagation.

---

### 2. Aplicación sobre Student Performance

Posteriormente se aplicó el mismo enfoque al dataset Student Performance.

En este caso el problema corresponde a una regresión, ya que se busca predecir la variable `Exam_Score`.

Para este problema se utilizó el error cuadrático medio (MSE) como función de aptitud.

Los datos numéricos fueron estandarizados y las variables categóricas fueron transformadas mediante One-Hot Encoding.

También se realizaron pruebas con diferentes cantidades de iteraciones de PSO para observar el comportamiento de la convergencia.

Los resultados se pueden observar en los gráficos generados por el programa:

- `convergencia_pso_student.png`
- `predicho_vs_real_student.png`

---

### 3. Clustering mediante PSO

También se implementó PSO para resolver un problema de clustering utilizando el dataset Student Performance.

En este caso, cada partícula representa un conjunto completo de 3 centroides.

Se utilizaron:

- 30 partículas.
- 3 clusters.
- 100 iteraciones.

La función de aptitud corresponde a la inercia, calculada como la suma de las distancias cuadráticas de cada estudiante al centroide más cercano.

El objetivo de PSO es minimizar esta inercia.

Es importante mencionar que `Exam_Score` no se utilizó como característica para crear los clusters. Esta variable se utilizó posteriormente para interpretar los grupos obtenidos.

Los resultados obtenidos fueron:

- Cluster 0: promedio de Exam_Score = 66.86
- Cluster 1: promedio de Exam_Score = 67.06
- Cluster 2: promedio de Exam_Score = 67.78

También se realizó una comparación con K-Means utilizando la inercia y el Silhouette Score.

Los gráficos generados son:

- `convergencia_pso_clustering.png`
- `clusters_pso_pca.png`

---

## 4. Archivos

| Archivo | Descripción |
|---|---|
| `pso_nn_training.py` | Entrenamiento de la red neuronal con PSO sin backpropagation |
| `pso_nn_student.py` | Aplicación de PSO a Student Performance |
| `pso_clustering.py` | Clustering mediante PSO |
| `convergencia_pso.png` | Convergencia del entrenamiento en Iris |
| `convergencia_pso_student.png` | Convergencia de PSO en Student Performance |
| `predicho_vs_real_student.png` | Comparación entre valores reales y predichos |
| `convergencia_pso_clustering.png` | Convergencia del PSO para clustering |
| `clusters_pso_pca.png` | Visualización de los clusters mediante PCA |
