# Algoritmos-de-enjambre-en-el-aprendizaje-autom-tico.



# Actividad 03 – Algoritmos de Enjambre

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
