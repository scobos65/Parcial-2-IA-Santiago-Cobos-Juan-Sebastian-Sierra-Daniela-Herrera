# Parcial-2-IA-Santiago-Cobos-Juan-Sebastian-Sierra-Daniela-Herrera
# Predicción de Ingresos con el Dataset Adult Census Income — Taller de Inteligencia Artificial Aplicada a la Economía

## Introducción

Este proyecto tiene como propósito aplicar herramientas de **aprendizaje automático** para predecir la probabilidad de que un individuo perciba ingresos superiores a 50.000 dólares anuales, a partir de variables socioeconómicas recogidas en el **Adult Census Income Dataset**. Desde la perspectiva de la economía aplicada, este ejercicio es relevante pues permite comprender cómo características demográficas, educativas y laborales se asocian con la probabilidad de situarse en un determinado tramo de ingresos. Además, constituye un caso práctico de cómo la inteligencia artificial puede apoyar el análisis empírico y la toma de decisiones en política pública y economía laboral.

---

## 1. Procesamiento de los datos

### 1.1 Carga y limpieza

Los archivos `adult.data`, `adult.test` y `adult.names` fueron empleados. Se extrajeron los nombres de las variables desde la documentación oficial y se unificó la variable objetivo (`income`) para garantizar consistencia entre las muestras de entrenamiento y prueba. Los valores faltantes se imputaron con la categoría **Unknown** en las variables laborales y de país de origen.

### 1.2 División en conjuntos

El conjunto de prueba original se dividió equitativamente en **validación** y **prueba final**, con estratificación según la variable objetivo. Esto asegura un equilibrio en la proporción de clases y permite evaluar el desempeño de los modelos de manera independiente y rigurosa.

### 1.3 Análisis exploratorio

Se examinaron tipos de datos, estadísticos descriptivos y distribuciones. Un hallazgo central fue el **desbalance de clases**: la mayoría de individuos percibe ingresos inferiores a 50.000 dólares, lo cual justifica la utilización de técnicas de ponderación de clases durante el entrenamiento.

### 1.4 Transformación de variables

* **Variable dependiente:** se codificó como binaria (`0` para `<=50K`, `1` para `>50K`).
* **Variables categóricas:** transformadas mediante **One-Hot Encoding**.
* **Variables numéricas:** normalizadas con **StandardScaler**.

Este preprocesamiento garantiza que los algoritmos puedan interpretar las variables de forma adecuada, evitando sesgos numéricos y preservando la naturaleza discreta de las categorías.

---

## 2. Modelos

### 2.1 Regresión logística (baseline)

La **regresión logística** se utilizó como modelo de referencia. El estimador fue configurado con penalización L2 (*ridge*), un parámetro de regularización $C=1.0$, y el solucionador **liblinear** para mayor eficiencia. Este modelo proporciona una primera aproximación lineal al problema y permite comparar posteriores mejoras no lineales.

### 2.2 Redes neuronales (MLP)

Se implementaron **perceptrones multicapa (MLP)** en PyTorch. La arquitectura consistió en capas densas totalmente conectadas con activación **ReLU**. El número de capas ocultas y neuronas por capa se trató como hiperparámetro. La capa de salida es una neurona lineal para clasificación binaria.

#### 2.2.1 Función de pérdida y optimización

La función de pérdida utilizada fue **BCEWithLogitsLoss**, la cual combina de manera estable la función logística con la entropía cruzada binaria. Para enfrentar el desbalance, se incluyó un parámetro `pos_weight`, calculado como la proporción inversa de observaciones positivas. El optimizador elegido fue **Adam**, por su capacidad de converger de manera eficiente con mínima necesidad de ajuste fino.

#### 2.2.2 Hiperparámetros y experimentos

Los principales hiperparámetros analizados fueron:

* **Número de capas ocultas y tamaño de cada capa**: desde arquitecturas compactas hasta profundas y anchas.
* **Tasa de aprendizaje (lr):** entre 0.001 y 0.0007.
* **Número de épocas:** entre 50 y 200.
* **Batch size:** 128.
* **Dropout:** tasas de 0.2 a 0.5 según la capa, para controlar el sobreajuste.
* **Early stopping:** con paciencia de 15 épocas, para detener el entrenamiento cuando la pérdida de validación deja de mejorar.
* **Regularización L2 (weight decay):** fijada en 1e-5.

#### 2.2.3 Resultados de los experimentos

* **Modelos sin regularización:** mostraron mejoras sobre la regresión logística, pero con claras señales de sobreajuste en arquitecturas profundas y anchas.
* **Modelos con regularización (Dropout + EarlyStopping):** alcanzaron un mejor equilibrio entre pérdida de entrenamiento y validación, mostrando mayor capacidad de generalización.
* El mejor modelo final se seleccionó según la **menor pérdida de validación** y se evaluó en los tres conjuntos (entrenamiento, validación, prueba).

---

## 3. Métricas de evaluación

Se utilizaron las siguientes métricas, relevantes en economía aplicada debido al interés en minimizar falsos positivos y falsos negativos de forma diferenciada:

* **Accuracy:** proporción de clasificaciones correctas.
* **Precisión:** utilidad en contextos donde los falsos positivos son costosos.
* **Recall (sensibilidad):** importancia en contextos donde no detectar ingresos altos puede implicar sesgos en política social.
* **F1 Score:** balance entre precisión y recall.
* **ROC-AUC:** medida robusta frente al desbalance de clases.

Los resultados mostraron que los MLP regularizados superan consistentemente a la regresión logística en términos de recall y F1, sin sacrificar la interpretabilidad básica que ofrece el modelo lineal.

---

## 4. Conclusiones

1. La regresión logística constituye un **baseline sólido** y de fácil interpretación económica, pero insuficiente para capturar relaciones no lineales.
2. Las redes neuronales (MLP), al incorporar hiperparámetros adecuados y técnicas de regularización, logran **mayor capacidad predictiva** y generalización.
3. Desde una perspectiva económica, el ejercicio demuestra el valor de la inteligencia artificial para complementar los métodos econométricos tradicionales, especialmente en problemas con estructuras de datos complejas y no lineales.
