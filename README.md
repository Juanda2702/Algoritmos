# Sistema de Predicción de Rendimiento y Retención Estudiantil

Este repositorio contiene el desarrollo práctico de un sistema predictivo basado en Aprendizaje Supervisado para el análisis del rendimiento académico y la identificación temprana de estudiantes en riesgo de deserción o reprobación. El proyecto implementa dos enfoques fundamentales de Machine Learning: Regresión Lineal Múltiple y Clasificación mediante Árboles de Decisión.

## 📋 Contexto del Caso Práctico
La retención escolar es un factor crítico en las instituciones educativas. Este sistema procesa datos históricos de comportamiento estudiantil (asistencia, entregas y uso de plataformas virtuales) con un doble propósito:
1. **Regresión:** Estimar con precisión la calificación final de un estudiante (escala 0-100).
2. **Clasificación:** Catalogar de forma binaria si un alumno se encuentra "En Riesgo" (1) o "Sin Riesgo" (0) para activar protocolos de tutoría preventiva.

## 📁 Estructura del Repositorio (Rama: dev)
* `modelos_estudiantes.py`: Script principal en Python que contiene el pipeline completo de datos (generación sintética basada en distribución normal, preprocesamiento, escalamiento, entrenamiento de modelos y evaluación).
* `README.md`: Documentación técnica del proyecto (este archivo).

## 🛠️ Requisitos e Instalación
Para ejecutar el script localmente, es necesario contar con un entorno de Python 3.8 o superior y tener instaladas las siguientes dependencias de ciencia de datos:

```bash
pip install pandas scikit-learn numpy
```
## 🚀 Ejecución del Modelo
Para ejecutar el pipeline de entrenamiento y desplegar las métricas de rendimiento en la consola, ejecute el siguiente comando en la terminal:

```bash
python modelos_estudiantes.py
