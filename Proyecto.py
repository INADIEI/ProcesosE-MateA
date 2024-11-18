import pandas as pd
import numpy as np


def limpiar_datos(df, nombre):
    
    print(f"\nLimpieza de la base de datos: {nombre}")
    
    # Dimensiones iniciales
    print(f"Dimensiones iniciales: {df.shape}")
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    # Verificar valores faltantes
    print("Valores faltantes por columna antes de limpieza:")
    print(df.isnull().sum())
    
    # Eliminar filas con valores faltantes
    df = df.dropna()
    
    # Dimensiones después de limpieza
    print(f"Dimensiones después de eliminar duplicados y valores faltantes: {df.shape}")
    
    print("Nombres de columnas después de normalización:")
    print(df.columns)
    
    return df


def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos geográficos en kilómetros.
    :param lat1: Latitud del punto 1
    :param lon1: Longitud del punto 1
    :param lat2: Latitud del punto 2
    :param lon2: Longitud del punto 2
    :return: Distancia en kilómetros
    """
    R = 6371  # Radio de la Tierra en kilómetros
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  # Convertir grados a radianes
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

## Funcion creada con base a la base de datos
def F(x, y, z):
    return ((x-2)**2 + 1) 
    ##return ((43738*x) + (1155*y) + (179428*z))

## Minimizacion con el metodo Dicotomico
def dicotomico (a, b, e = 0.01):
    resFinal = 0
    i = 0
    
    ## Pienso que deberia ser el numero promedio de los datos de la base de datos
    res = 1000000000000000000000
    while(abs(b - a) > e):
        x1 = ((a+b)/2) - e
        x2 = ((a+b)/2) + e
                
        if (F(x1,x1,x1)<F(x2,x2,x2)):
            b = x2
            res = F(x1,x1,x1)
        else:
            a = x1
            res = F(x2,x2,x2)

        if (res<F(((a+b)/2) - e,((a+b)/2) - e,((a+b)/2) - e) and res<F(((a+b)/2) + e,((a+b)/2) + e,((a+b)/2) + e)):
            if(res<F(((a+b)/2) - e,((a+b)/2) - e,((a+b)/2) - e)):
                resFinal = x1
                break
            else:
                resFinal = x2
                break
            
        print("esta es la iteracion: ", i)
        print(res, x1, x2)
        i+=1
    return resFinal

# Implementación de cadenas de Markov
def Markov(matriz, states, steps):

    c = np.random.choice(states)
    v = [c]

    for _ in range(steps):
        c = np.random.choice(states, p=matriz[states.index(c)])
        v.append(c)

    return v

# Optimización logística con distancia y tiempo
def optimizacion(df, distancia_col, tiempo_col, costo_distancia, costo_tiempo):
    
    df['Costo_total'] = df[distancia_col] * costo_distancia + df[tiempo_col] * costo_tiempo
    costo_total = df['Costo_total'].sum()
    return costo_total

            

resultadoD = dicotomico(0,10)

print(resultadoD)


uber = pd.read_csv('My Uber Drives - 2016.csv')
amazon = pd.read_csv('amazon_delivery.csv')
dataset = pd.read_csv('dataset.csv')  ## No tiene Distancias, solo tiempo entre recogida y entrega

# Limpiar cada base de datos
uber_limpio = limpiar_datos(uber, "Uber")
amazon_limpio = limpiar_datos(amazon, "Amazon")
dataset_limpio = limpiar_datos(dataset, "Dataset")


# Calcular la distancia para cada fila del DataFrame
amazon_limpio['Distancia_km'] = haversine(
    amazon_limpio['Store_Latitude'], amazon_limpio['Store_Longitude'],
    amazon_limpio['Drop_Latitude'], amazon_limpio['Drop_Longitude']
)

# Cadenas de Markov ejemplo
transition_matrix = [
    [0.7, 0.3],  # Probabilidades desde el estado 0
    [0.4, 0.6]   # Probabilidades desde el estado 1
]
states = ['Estado 0', 'Estado 1']
markov_simulation = Markov(transition_matrix, states, 10)
print("Simulación de cadenas de Markov:", markov_simulation)

# Optimización logística
costo_optimizado = optimizacion(amazon_limpio, 'Distancia_km', 'Delivery_Time', 5, 2)
print(f"Costo logístico optimizado: {costo_optimizado}")

# Verificar los resultados
print(amazon_limpio[['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Distancia_km']])

disUber = uber_limpio['MILES*'] ## esto es en millas
print(disUber)
