import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


## Modelo de Regresion Lineal con Ruido Estocastico

# Simulación de datos
np.random.seed(42)
distancia = np.linspace(1, 100, 1000)  # Distancia en kilómetros
demanda = 10 + 0.5 * distancia + np.random.normal(0, 5, size=1000)  # Relación lineal con ruido

# Graficar los datos simulados
plt.scatter(distancia, demanda, alpha=0.5, label='Datos simulados')
plt.xlabel("Distancia (km)")
plt.ylabel("Demanda")
plt.legend()
plt.show()

# Ajustar un modelo de regresión lineal
from sklearn.linear_model import LinearRegression

X = distancia.reshape(-1, 1)  # Redimensionar para sklearn
modelo = LinearRegression()
modelo.fit(X, demanda)

# Predicciones
predicciones = modelo.predict(X)

# Graficar resultados
plt.scatter(distancia, demanda, alpha=0.5, label='Datos simulados')
plt.plot(distancia, predicciones, color='red', label='Modelo ajustado')
plt.xlabel("Distancia (km)")
plt.ylabel("Demanda")
plt.legend()
plt.show()


## Modelo de Simulacion de Monte Carlo

# Supongamos que la relación distancia-demanda tiene una desviación estándar fija
def simular_demanda(distancia, n_simulaciones=1000):
    # Coeficientes de la relación
    intercepto = 10
    coef_distancia = 0.5
    ruido_std = 5
    
    simulaciones = []
    for _ in range(n_simulaciones):
        demanda_simulada = intercepto + coef_distancia * distancia + np.random.normal(0, ruido_std, size=len(distancia))
        simulaciones.append(demanda_simulada)
    
    return np.array(simulaciones)

# Simular 1000 escenarios
simulaciones = simular_demanda(distancia)

# Calcular media y percentiles
media_demanda = np.mean(simulaciones, axis=0)
p5 = np.percentile(simulaciones, 5, axis=0)
p95 = np.percentile(simulaciones, 95, axis=0)

# Graficar resultados
plt.plot(distancia, media_demanda, label='Demanda media', color='blue')
plt.fill_between(distancia, p5, p95, color='blue', alpha=0.2, label='Intervalo 5%-95%')
plt.scatter(distancia, demanda, alpha=0.3, label='Datos simulados')
plt.xlabel("Distancia (km)")
plt.ylabel("Demanda")
plt.legend()
plt.show()


## Modelo ARIMA Estocastico para Series Temporales

# Crear una serie temporal simulada
np.random.seed(42)
fechas = pd.date_range(start="2023-01-01", periods=100, freq='D')
distancia = np.linspace(1, 50, 100)  # Distancia diaria
demanda = 10 + 0.5 * distancia + np.random.normal(0, 3, size=100)  # Relación lineal con ruido

# Crear un DataFrame
df = pd.DataFrame({'fecha': fechas, 'demanda': demanda})
df.set_index('fecha', inplace=True)

# Ajustar un modelo ARIMA
modelo_arima = ARIMA(df['demanda'], order=(2, 1, 2))
resultado = modelo_arima.fit()

# Predicción
predicciones = resultado.forecast(steps=20)

# Graficar resultados
plt.plot(df.index, df['demanda'], label='Demanda real')
plt.plot(pd.date_range(start=df.index[-1], periods=20, freq='D'), predicciones, label='Predicción', color='red')
plt.xlabel("Fecha")
plt.ylabel("Demanda")
plt.legend()
plt.show()

## Modelo de Cadena de Markov

# Definir los estados y la matriz de transición
estados = ['Baja', 'Media', 'Alta']
matriz_transicion = np.array([[0.6, 0.3, 0.1],
                              [0.2, 0.5, 0.3],
                              [0.1, 0.3, 0.6]])

# Simular una cadena de Markov
def cadena_markov(matriz, estados, n_pasos=50, estado_inicial='Media'):
    estado_actual = estados.index(estado_inicial)
    cadena = [estado_inicial]
    
    for _ in range(n_pasos):
        estado_actual = np.random.choice(len(estados), p=matriz[estado_actual])
        cadena.append(estados[estado_actual])
    
    return cadena

# Simulación
cadena_simulada = cadena_markov(matriz_transicion, estados, n_pasos=100)

# Graficar la simulación
plt.plot(cadena_simulada, marker='o', label='Demanda simulada')
plt.xlabel("Paso")
plt.ylabel("Estado de demanda")
plt.legend()
plt.show()
