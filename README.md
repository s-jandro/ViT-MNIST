# Vision Transformer (ViT) "From Scratch" en C++ para MNIST


Este repositorio contiene una implementación completa y manual de un **Vision Transformer (ViT)** escrita en **C++ puro**, diseñada para resolver la tarea de clasificación de dígitos manuscritos (dataset MNIST).
## 👤 Autores

- **Sergio Alejandro Paucar Cruz**
- **Samuel Alexander Iman Quispe**
- **Renato Oscar Corrales Peña**

## 🚀 Características Principales

  * **Implementación "Desde Cero":** No se utilizan librerías de Deep Learning de alto nivel (como PyTorch, TensorFlow o Keras).
  * **Motor de Autograd Manual:** El mecanismo de *Backpropagation* (propagación hacia atrás), incluyendo las derivadas de la *Self-Attention*, *Layer Normalization* y las capas lineales, ha sido implementado manualmente.
  * **Matemática con Eigen:** Se utiliza la librería `Eigen` exclusivamente para operaciones eficientes de álgebra lineal (multiplicación de matrices, etc.).
  * **Entrenamiento e Inferencia:** El sistema permite entrenar el modelo desde cero con el dataset MNIST y realizar predicciones sobre nuevas imágenes externas.
  * **Persistencia del Modelo:** Capacidad para guardar y cargar los pesos entrenados (`.bin`) para evitar re-entrenar.
  * **Carga de Imágenes Propias:** Integración con `stb_image.h` para cargar y procesar imágenes JPG/PNG dibujadas por el usuario para pruebas en vivo.

## 🛠️ Arquitectura del Proyecto

Basado en la estructura de archivos actual:

```
VIT-MNIST/
├── Eigen/                     # Librería de álgebra lineal (dependencia)
├── stb_image.h                # Librería de un solo archivo para cargar imágenes
├── main2.cpp                  # Código fuente principal (modelo, entrenamiento, menú)
├── main2.exe                  # Ejecutable compilado
│
├── Datos MNIST (Dataset):
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
│
├── Archivos Generados:
│   ├── vit_mnist_weights.bin  # Pesos del modelo entrenado (GUARDAR ESTO)
│   └── Datos ultimo entrenamiento.txt # Logs de entrenamiento
│
└── numero.jpg                 # Imagen de ejemplo para pruebas de predicción
```

## 📋 Requisitos Previos

  * Un compilador de C++ compatible con estándares modernos (GCC, Clang, MSVC). Se recomienda usar flags de optimización (`-O3`).
  * Los archivos del dataset MNIST (incluidos en este repositorio).
  * La carpeta `Eigen` y el archivo `stb_image.h` (incluidos en este repositorio).

## ⚙️ Compilación

Para compilar el proyecto, asegúrate de que el compilador pueda encontrar la carpeta `Eigen`. Un comando de ejemplo usando `g++` sería:

```bash
g++ -I. main2.cpp -O3 -o main2.exe
```

*(Nota: `-I.` indica al compilador que busque archivos de cabecera en el directorio actual, necesario para encontrar `Eigen/Dense` y `stb_image.h`).*

## 💻 Uso

Al ejecutar el programa (`./main2.exe`), aparecerá un menú interactivo en la consola:

### Opción 1: Entrenar modelo (Train)

  * Inicia el proceso de entrenamiento sobre las 60,000 imágenes de MNIST.
  * Te preguntará si deseas continuar un entrenamiento previo (cargando `vit_mnist_weights.bin`) o empezar desde cero.
  * Muestra el progreso y la precisión en tiempo real.
  * Guarda automáticamente los pesos al finalizar cada época.

### Opción 2: Probar imagen propia (Predict)

  * Carga el modelo entrenado (`vit_mnist_weights.bin`). **Debes haber entrenado al menos una vez antes de usar esta opción.**
  * Te pedirá la ruta de una imagen (por ejemplo, `numero.jpg`).
  * La imagen será preprocesada (invertida y normalizada) para ajustarse al formato MNIST.
  * El modelo mostrará la predicción del dígito y su nivel de confianza.

-----


