# Optimización de hiperparámetros de modelos de redes neuronales para la estimación de PM₂.₅

Este repositorio contiene los scripts desarrollados para la **optimización, entrenamiento, evaluación y comparación de diferentes arquitecturas de redes neuronales** empleadas en la estimación de concentraciones de material particulado fino (**PM₂.₅**) en la ciudad de **Aguascalientes, México**.

La optimización de hiperparámetros se realiza mediante **Optuna**, mientras que la construcción y entrenamiento de los modelos se lleva a cabo utilizando **TensorFlow/Keras**.

El repositorio también incluye herramientas para la obtención y procesamiento de algunas **variables espaciales asociadas al entorno urbano**, utilizando información proveniente del **Instituto Nacional de Estadística y Geografía (INEGI)** y **OpenStreetMap (OSM)**.

---

## Modelos evaluados

Se consideran cinco arquitecturas de redes neuronales:

* **FNN** — Feedforward Neural Network
* **RNN** — Simple Recurrent Neural Network
* **LSTM** — Long Short-Term Memory
* **GRU** — Gated Recurrent Unit
* **CNN** — One-dimensional Convolutional Neural Network (1D-CNN)

Los modelos utilizan información ambiental, temporal y espacial para estimar la concentración de PM₂.₅.

---

## Variables de entrada

Los scripts de entrenamiento consideran variables como:

* Concentración previa de PM₂.₅
* Temperatura (`TMP`)
* Latitud
* Longitud
* Componentes de velocidad del viento (`WSx`, `WSy`)
* Densidad poblacional
* Variables temporales transformadas mediante funciones seno y coseno:

  * Hora del día
  * Día de la semana
  * Día del año

Las variables son normalizadas al intervalo `[-1, 1]` mediante `MinMaxScaler`.

---

## Optimización de hiperparámetros

La optimización se realiza utilizando **Optuna** y su algoritmo `TPESampler`.

Dependiendo de la arquitectura, se exploran diferentes combinaciones de hiperparámetros, incluyendo:

* Número de neuronas o unidades
* Número de capas
* Función de activación
* Optimizador
* Tasa de aprendizaje
* Tamaño de batch
* Función de pérdida
* Número de pasos temporales (`time_step`)
* Número de filtros convolucionales para CNN

Entre las funciones de activación consideradas se encuentran:

* ReLU
* Sigmoid
* Tanh
* Leaky ReLU

Los optimizadores evaluados incluyen:

* Adam
* SGD
* RMSprop

Las funciones de pérdida evaluadas incluyen:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)

Cada estudio de Optuna busca minimizar el error obtenido durante la validación del modelo.

---

## Estructura del repositorio

```text
.
├── opti_FNN.py
├── opti_RNN.py
├── opti_LSTM.py
├── opti_GRU.py
├── opti_CNN.py
│
├── eval_models.py
├── anova_modelos_rn.py
│
├── datos_INEGI.py
├── densidad_po_INEGI.py
├── densidad_industrias.py
├── densidad_rutas.py
│
└── README.md
```

### Scripts de optimización

#### `opti_FNN.py`

Optimización de hiperparámetros de una red neuronal feedforward (FNN).

El script explora diferentes configuraciones de arquitectura, funciones de activación, optimizadores, tasas de aprendizaje, tamaños de batch y funciones de pérdida.

---

#### `opti_RNN.py`

Optimización de una red neuronal recurrente simple (`SimpleRNN`).

Además de los hiperparámetros asociados al entrenamiento, se evalúa la longitud de las secuencias temporales utilizadas como entrada al modelo.

---

#### `opti_LSTM.py`

Optimización de una arquitectura **Long Short-Term Memory (LSTM)** para considerar las dependencias temporales presentes en las concentraciones de PM₂.₅.

---

#### `opti_GRU.py`

Optimización de una arquitectura **Gated Recurrent Unit (GRU)**.

Se evalúan diferentes números de unidades recurrentes, funciones de activación, optimizadores, tasas de aprendizaje, tamaños de batch, funciones de pérdida y longitudes de secuencia temporal.

---

#### `opti_CNN.py`

Optimización de una arquitectura **CNN 1D** aplicada a secuencias temporales.

La arquitectura incluye:

```text
Conv1D
  ↓
MaxPooling1D
  ↓
Flatten
  ↓
Dense
  ↓
Output
```

Entre los hiperparámetros explorados se encuentran el número de filtros convolucionales y el número de unidades de la capa densa.

---

## Evaluación de los modelos

### `eval_models.py`

Entrena nuevamente los modelos utilizando las mejores configuraciones identificadas mediante Optuna.

Se realizan múltiples repeticiones del entrenamiento para obtener una distribución del error de cada arquitectura y permitir una comparación estadística entre los modelos.

---

### `anova_modelos_rn.py`

Realiza la comparación estadística del desempeño de las arquitecturas utilizando los valores de MSE obtenidos durante los diferentes entrenamientos.

El análisis incluye:

1. **ANOVA de una vía**, para determinar si existen diferencias estadísticamente significativas entre los modelos.

2. **Prueba post hoc de Tukey HSD**, para identificar específicamente qué pares de modelos presentan diferencias significativas.

Los modelos comparados son:

```text
FNN
RNN
LSTM
GRU
CNN
```

---

## Variables espaciales

Además de los modelos de redes neuronales, el repositorio incluye scripts auxiliares para generar variables relacionadas con las características urbanas alrededor de un punto geográfico.

### `densidad_po_INEGI.py`

Obtiene información de población por manzana mediante servicios geográficos del **INEGI**.

Para un punto definido por latitud y longitud, se construye un buffer y se calcula la población contenida dentro de éste considerando la fracción del área de cada manzana que intersecta el área de estudio.

La variable resultante se expresa como:

```text
habitantes / km²
```

---

### `densidad_industrias.py`

Obtiene información de posibles fuentes industriales utilizando **OpenStreetMap** mediante la librería `OSMnx`.

Se consideran elementos asociados con etiquetas como:

```text
landuse = industrial
building = industrial
building = warehouse
building = factory
industrial = *
man_made = works
```

A partir de los elementos encontrados dentro de un buffer se calcula una densidad de fuentes industriales.

---

### `densidad_rutas.py`

Obtiene la red vial mediante **OpenStreetMap/OSMnx** y calcula la longitud de vialidades contenida dentro de un radio determinado.

El script permite seleccionar diferentes categorías de vialidad, incluyendo carreteras primarias, secundarias, residenciales y otras clasificaciones de OpenStreetMap.

La densidad vial se expresa como:

```text
km de vialidad / km²
```

---

### `datos_INEGI.py`

Script utilizado para explorar y procesar información geográfica y poblacional obtenida del INEGI.

Incluye herramientas para:

* Consulta de información por manzana.
* Cálculo de áreas.
* Estimación de densidad poblacional.
* Representación espacial de los datos.
* Interpolación mediante **Inverse Distance Weighting (IDW)**.

---

## Resultados generados por Optuna

Los scripts de optimización permiten generar diferentes herramientas para analizar el proceso de búsqueda de hiperparámetros, entre ellas:

### Historial de optimización

Permite observar la evolución de la función objetivo conforme aumenta el número de ensayos de Optuna.

### Importancia de hiperparámetros

Permite identificar cuáles hiperparámetros tienen mayor influencia sobre el desempeño del modelo.

### Coordenadas paralelas

Permiten visualizar conjuntamente las combinaciones de hiperparámetros exploradas durante la optimización.

Los resultados de cada estudio también pueden exportarse como tablas para su posterior análisis y documentación.

---

## Requisitos

El proyecto utiliza principalmente las siguientes librerías de Python:

```text
numpy
pandas
tensorflow
keras
scikit-learn
optuna
matplotlib
scipy
statsmodels
requests
geopandas
shapely
osmnx
contextily
```

Pueden instalarse mediante `pip`, por ejemplo:

```bash
pip install numpy pandas tensorflow scikit-learn optuna matplotlib scipy statsmodels requests geopandas shapely osmnx contextily
```

Se recomienda utilizar un entorno virtual para mantener aisladas las dependencias del proyecto.

---

## Datos de entrada

Los scripts de optimización utilizan una base de datos denominada:

```text
base_datos_mod.csv
```

Entre las columnas utilizadas se encuentran:

```text
PM25
TMP
latitud
longitud
sin_hour
cos_hour
sin_dow
cos_dow
sin_doy
cos_doy
WSx
WSy
den_po
```

El archivo de datos no necesariamente se incluye en este repositorio debido a las características y fuentes originales de la información utilizada.

---

## Configuración

Antes de ejecutar los scripts es necesario modificar las rutas locales definidas dentro de algunos archivos.

Por ejemplo:

```python
ruta = "ruta/al/directorio/de/trabajo/"
```

Estas rutas corresponden al entorno utilizado durante el desarrollo original y deben sustituirse de acuerdo con la ubicación de los archivos en el equipo del usuario.

---

## Ejecución

Por ejemplo, para optimizar la arquitectura GRU:

```bash
python opti_GRU.py
```

Para optimizar una LSTM:

```bash
python opti_LSTM.py
```

Para evaluar los modelos con los mejores hiperparámetros:

```bash
python eval_models.py
```

Y para realizar la comparación estadística:

```bash
python anova_modelos_rn.py
```

---

## Flujo general de trabajo

```text
Datos ambientales y espaciales
            │
            ▼
     Preprocesamiento
            │
            ▼
   Normalización de datos
            │
            ▼
 Creación de secuencias temporales
            │
            ▼
 ┌───────────────────────────┐
 │ Optimización con Optuna   │
 └───────────────────────────┘
            │
            ▼
 ┌───────────────────────────┐
 │ FNN │ RNN │ LSTM │ GRU │ CNN │
 └───────────────────────────┘
            │
            ▼
 Mejores hiperparámetros
            │
            ▼
 Entrenamientos repetidos
            │
            ▼
 Comparación del desempeño
            │
            ▼
       ANOVA + Tukey HSD
```

---

## Fuentes de información geoespacial

Las variables espaciales utilizadas por algunos de los scripts se construyen a partir de información pública proveniente de:

* **INEGI** — Instituto Nacional de Estadística y Geografía.
* **OpenStreetMap (OSM)**.

La disponibilidad y cobertura de los datos de OpenStreetMap puede variar espacial y temporalmente.

---

## Aplicación

Estos códigos fueron desarrollados como parte de trabajos de investigación relacionados con la aplicación de **aprendizaje automático y redes neuronales a problemas de calidad del aire**, particularmente para la estimación de concentraciones de **PM₂.₅ en Aguascalientes, México**.

El repositorio tiene como objetivo facilitar la documentación y reproducibilidad de los procedimientos utilizados para:

* Construcción de variables ambientales y espaciales.
* Optimización de redes neuronales.
* Evaluación de modelos.
* Comparación estadística de arquitecturas.

---

## Autor

**Héctor Antonio Olmos Guerrero**

Research in environmental modeling, air quality and machine learning.

---

## Licencia

Este repositorio se proporciona con fines académicos y de investigación.

La licencia específica puede definirse de acuerdo con las condiciones de uso y distribución deseadas para el código.
