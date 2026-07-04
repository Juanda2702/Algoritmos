import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report

print("--- INICIANDO PIPELINE DE MACHINE LEARNING ---")

# 1. GENERACIÓN DE DATOS SIMULADOS (Para tener datos con los cuales trabajar)
np.random.seed(42)
n_samples = 2000

asistencia = np.random.normal(80, 15, n_samples).clip(0, 100)
horas_plataforma = np.random.normal(10, 4, n_samples).clip(0, 40)
tareas = np.random.normal(7, 2, n_samples).clip(0, 10)

# Calculamos una calificación final lógica basada en las variables anteriores con un poco de ruido
calificacion_final = (asistencia * 0.4) + (horas_plataforma * 1.5) + (tareas * 4) + np.random.normal(0, 5, n_samples)
calificacion_final = calificacion_final.clip(0, 100)

# Definimos el riesgo: 1 (En Riesgo) si la calificación es menor a 70, 0 si es mayor o igual
estado_riesgo = np.where(calificacion_final < 70, 1, 0)

# Creamos el DataFrame
df = pd.DataFrame({
    'asistencia': asistencia,
    'horas_plataforma': horas_plataforma,
    'tareas_entregadas': tareas,
    'calificacion_final': calificacion_final,
    'estado_riesgo': estado_riesgo
})

print("Datos generados correctamente. Total de alumnos:", len(df))
print("-" * 50)

# ==========================================
# MODELO 1: REGRESIÓN LINEAL (Calificación)
# ==========================================
X_reg = df[['asistencia', 'horas_plataforma', 'tareas_entregadas']]
y_reg = df['calificacion_final']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train_r_scaled, y_train_r)
predicciones_reg = modelo_regresion.predict(X_test_r_scaled)

print("\n--- RESULTADOS DEL MODELO DE REGRESIÓN MÚLTIPLE ---")
print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test_r, predicciones_reg):.2f}")
print(f"Error Absoluto Medio (MAE): {mean_absolute_error(y_test_r, predicciones_reg):.2f} puntos")
print(f"Coeficiente de Determinación (R²): {r2_score(y_test_r, predicciones_reg):.4f}")


# ==========================================
# MODELO 2: ÁRBOL DE DECISIÓN (Riesgo)
# ==========================================
X_clf = df[['asistencia', 'horas_plataforma', 'tareas_entregadas']]
y_clf = df['estado_riesgo']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# max_depth=4 para evitar sobreajuste (optimización)
modelo_arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo_arbol.fit(X_train_c, y_train_c)
predicciones_clf = modelo_arbol.predict(X_test_c)

print("\n--- RESULTADOS DEL MODELO DE CLASIFICACIÓN (ÁRBOL DE DECISIÓN) ---")
print(f"Precisión General (Accuracy): {accuracy_score(y_test_c, predicciones_clf) * 100:.2f}%")
print("\nReporte Detallado:")
print(classification_report(y_test_c, predicciones_clf))