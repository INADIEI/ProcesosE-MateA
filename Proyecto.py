#Librerías
import pandas as pd
import numpy as np

class ProcesadorDatos:
    def __init__(self):
        pass
    
    def funciones_diccionario():
        print("Estas son las funciones dentro de la clase: \n")
        #1Limpieza de datos
        print("\nlimpiar_datos(df,nombre)")
        #2haversine
        print("\nhaversine(lat1,lon1,lat2,lon2)")
        #3F
        print("\nF(x,y,z)")
        #4dicotomico
        print("\ndicotomico(a,b,e=0.01)")
        #5Markov
        print("\nMarkov(matriz, states, steps)")
        

    def limpiar_datos(df, nombre):
        #Limpieza datos elimina duplicados y valores faltantes
        print(f"\nLimpieza de la base de datos: {nombre}")
        # Dimensiones iniciales
        print(f"Dimensiones iniciales: {df.shape}")
        df = df.drop_duplicates() # Eliminar duplicados
        # Verificar valores faltantes
        print("Valores faltantes por columna antes de limpieza:")
        print(df.isnull().sum())
        df = df.dropna() # Eliminar filas con valores faltantes
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

    def F(x, y, z): ## Funcion creada con base a la base de datos
        return ((x-2)**2 + 1) 
        ##return ((43738*x) + (1155*y) + (179428*z))


    def dicotomico (a, b, e = 0.01):
        ## Minimizacion con el metodo Dicotomico
        resFinal = 0
        i = 0
        res = float('inf')
        while(abs(b - a) > e):
            x1 = ((a+b)/2) - e
            x2 = ((a+b)/2) + e
                
            if DataProcessor.F(x1, x1, x1) < DataProcessor.F(x2, x2, x2):
                b = x2
                res = DataProcessor.F(x1, x1, x1)
            else:
                a = x1
                res = DataProcessor.F(x2, x2, x2)
            if res < DataProcessor.F(((a + b) / 2) - e, ((a + b) / 2) - e, ((a + b) / 2) - e):
                resFinal = x1
                break
            elif res < DataProcessor.F(((a + b) / 2) + e, ((a + b) / 2) + e, ((a + b) / 2) + e):
                resFinal = x2
                break
            print("Iteración:", i, "Resultado:", res, "x1:", x1, "x2:", x2)
            i += 1
        return resFinal
    
    def simplex ():

    def Markov(matriz, states, steps):
        # Implementación de cadenas de Markov
        c = np.random.choice(states)
        v = [c]
        for _ in range(steps):
            c = np.random.choice(states, p=matriz[states.index(c)])
            v.append(c)
        return v

def optimizacion(df, distancia_col, tiempo_col, costo_distancia, costo_tiempo):
        """
        Optimización logística considerando distancia y tiempo.
        """
        df['Costo_total'] = df[distancia_col] * costo_distancia + df[tiempo_col] * costo_tiempo
        costo_total = df['Costo_total'].sum()
        return costo_total

            

    #resultadoD = dicotomico(0,10)

    #print(resultadoD)


    #uber = pd.read_csv('DataSets/My Uber Drives - 2016.csv')
    #amazon = pd.read_csv('DataSets/amazon_delivery.csv')
    #dataset = pd.read_csv('DataSets/dataset.csv')  ## No tiene Distancias, solo tiempo entre recogida y entrega

    # Limpiar cada base de datos
    #uber_limpio = limpiar_datos(uber, "Uber")
    #amazon_limpio = limpiar_datos(amazon, "Amazon")
    #dataset_limpio = limpiar_datos(dataset, "Dataset")


    # Calcular la distancia para cada fila del DataFrame
    #amazon_limpio['Distancia_km'] = haversine(
        #amazon_limpio['Store_Latitude'], amazon_limpio['Store_Longitude'],
        #amazon_limpio['Drop_Latitude'], amazon_limpio['Drop_Longitude']
    #)

    # Cadenas de Markov ejemplo
    #transition_matrix = [
        #[0.7, 0.3],  # Probabilidades desde el estado 0
        #[0.4, 0.6]   # Probabilidades desde el estado 1
    #]
    #states = ['Estado 0', 'Estado 1']
    #markov_simulation = Markov(transition_matrix, states, 10)
    #print("Simulación de cadenas de Markov:", markov_simulation)

    # Optimización logística
    #costo_optimizado = optimizacion(amazon_limpio, 'Distancia_km', 'Delivery_Time', 5, 2) ## El 5 y 2 son ejemplos de costo de distancia y tiempo, no de la base de datos
    #print(f"Costo logístico optimizado: {costo_optimizado}")

    # Verificar los resultados
    #print(amazon_limpio[['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Distancia_km']])

    #disUber = uber_limpio['MILES*'] ## esto es en millas
    #print(disUber)


class MetodoDicotomico:
    def __init__(self, df, epsilon = 0.01, tolerancia = 0.01):
        self.tiempo = df['Tiempo']
        self.distancia = df['Distancia']
        self.epsilon = epsilon
        self.tolerancia = tolerancia

    def f(self, x):
        v1 = np.sum(self.tiempo)**(1/2)
        v2 = np.sum(self.distancia)**(1/4)
        i = v2/v1
        c = (np.abs(v1-v2))**(1/8)
        d = i*x**2 - c*x
        return d
    
    def aplicarMetodo(self):
        self.tiempo = pd.to_timedelta(self.tiempo).dt.total_seconds()
        a = min(self.tiempo)
        b = max(self.tiempo)
        
        # Método dicotómico para encontrar el mínimo tiempo
        while (b - a) > self.tolerancia:
            x1 = (a + b) / 2 - self.epsilon
            x2 = (a + b) / 2 + self.epsilon

            ev_x1 = self.f(x1)
            ev_x2 = self.f(x2)

            if ev_x1 < ev_x2:
                b = x2
            else:
                a = x1

        a = pd.to_timedelta(a, unit='s')
        b = pd.to_timedelta(b, unit='s')
        self.intervalo = (a, b)

        return self.intervalo
    

    import numpy as np

class MetodoSimplex:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.solution = None
        self.optimal_value = None

    def _pivot(self, tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for i in range(tableau.shape[0]):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

    def aplicarMetodo(self):
        # Dimensiones de A
        m, n = self.A.shape

        # Construcción de la tabla inicial del simplex
        tableau = np.hstack([self.A, np.eye(m), self.b.reshape(-1, 1)])
        c_extended = np.hstack([self.c, np.zeros(m + 1)])
        tableau = np.vstack([tableau, c_extended])

        # Iterar hasta que no haya coeficientes negativos en la fila de costos
        while any(tableau[-1, :-1] < 0):
            # Selección de la columna pivote (regla de Bland)
            col = np.where(tableau[-1, :-1] < 0)[0][0]

            # Selección de la fila pivote
            row = np.argmin(tableau[:-1, -1] / tableau[:-1, col])

            # Realizamos la operación de pivoteo
            self._pivot(tableau, row, col)

        # La solución óptima
        solution = np.zeros(n)
        for i in range(n):
            if np.all(tableau[:-1, i] == 1) and tableau[-1, i] == 0:
                solution[i] = tableau[np.where(tableau[:-1, i] == 1)[0][0], -1]

        self.solution = solution
        self.optimal_value = -tableau[-1, -1]
        return self.solution, self.optimal_value

