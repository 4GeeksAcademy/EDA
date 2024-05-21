#from utils import db_connect
#engine = db_connect()

import pandas as pd

# ¿Qué quiero resolver?
'''
 id -> no es info de la que sacar conclusiones                     
 name                           
 host_id -> no es info de la que sacar conclusiones                        
 host_name                     
 neighbourhood_group -> ¿Cuales son las diferencias de precios entre grupos de barrios y barrios?¿Qué barrios son mas populares?           
 neighbourhood                  
 latitude -> ¿Hay áreas con una mayor densidad de propiedades?¿Cómo varía el precio y la disponibilidad según la ubicación geográfica?                       
 longitude                       
 room_type                      
 price -> se puede realizar un análisis del precio ¿qué barrios tienen precios mas altos y mas bajos? ¿Hay relación entre precio y disponibilidad?¿Cómo varían los precios según el tipo de habitación?                  
 minimum_nights                    
 number_of_reviews                
 last_review -> ¿existe alguna relación entre la disponibilidad, la última reseña y el precio?                   
 reviews_per_month        
 calculated_host_listings_count -> ¿Cuántas propiedades tiene cada anfitrión?¿qué anfitriones tienen las propiedades con mejores reseñas?
 availability_365 -> ¿cómo varía la disponibilidad según el tipo de habitación y barrio?
'''
#Voy a hacer análisis del precio
# 1. Cargar conjunto de datos
total_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv')
total_data.head()

# 2. Exploración y limpieza de datos
#   Dimensiones e información
total_data.shape #48895 filas y 16 columnas

# Información 
total_data.info() # Las columnas 1, 3, 12 y 13 tienen valores nulos

#   Eliminación de duplicados
if total_data.duplicated().sum():
    total_data = total_data.drop_duplicates()
print(total_data.shape)
total_data.head()
# No hay duplicados

#   Eliminación de información irrelevante 
total_data.drop(["id", "name", "host_id"], axis = 1, inplace = True)
total_data.head()

# 3. Análisis de variables univariante
import matplotlib.pyplot as plt 
import seaborn as sns

#   Variables categóricas -> histogramas
#       No todas las variables se pueden representar mediante histogramas por tener demasiados tipos de valores
name_counts = total_data['host_name'].value_counts()
print(name_counts) # Hay 11452 anfitriones
neighbourhood_counts = total_data['neighbourhood'].value_counts()
print(neighbourhood_counts) # Hay 221 barrios
fig, axis = plt.subplots(1, 2, figsize = (10, 5))

# Histograma múltiple
#"host_name" -> existen demasiados nombres como para crear hisograma

sns.histplot(ax = axis[0], data = total_data, x = "neighbourhood_group")
sns.histplot(ax = axis[1], data = total_data, x = "room_type")

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

# Conclusiones: 
#   - El barrio donde existen mas pisos anunciados es Manhattan seguido de Brooklyn. El que menos Staten Island
#   - La mayoría de pisos se alquilan enteros, seguidos de habitaciones privadas. Las habitaciones compartidas lo que menos

#   Variables numéricas -> histogramas + diagramas de cajas
fig, axis = plt.subplots(4, 2, figsize = (18, 15), gridspec_kw={'height_ratios': [6, 1, 6, 1]})

# Crear una figura múltiple con histogramas y diagramas de caja
sns.histplot(ax = axis[0, 0], data = total_data, x = "price") # Hay un montón de datos atípicos
sns.boxplot(ax = axis[1, 0], data = total_data, x = "price")
sns.histplot(ax = axis[0, 1], data = total_data, x = "minimum_nights") # Hay un montón de datos atípicos
sns.boxplot(ax = axis[1, 1], data = total_data, x = "minimum_nights")
sns.histplot(ax = axis[2, 0], data = total_data, x = "availability_365")
sns.boxplot(ax = axis[3, 0], data = total_data, x = "availability_365")
sns.histplot(ax = axis[2, 1], data = total_data, x = "reviews_per_month") # Hay datos atípicos raros
sns.boxplot(ax = axis[3, 1], data = total_data, x = "reviews_per_month")

# ¿Existen pisos cuya disponibilidad es de 0 días? -> los elimino porque para hacer un análisis del precio pueden meter ruido
no_dispo = total_data['availability_365'] == 0
suma_no_dispo = no_dispo.sum()
print(suma_no_dispo) # Hay 17533 pisos que no han tenido disponibilidad en todo el año. Esto es el 35%

# En las noches mínimas existen datos atípicos demasiado altos -> a estudiar cuando llegue s su punto
# Ajustar el layout
plt.tight_layout()
#fig.delaxes(axis[2, 1])
#fig.delaxes(axis[3, 1])

# Mostrar el plot
plt.show()

# Elimino los pisos de disponibilidad 0 al no aportar nada a un análisis de los precios
total_data = total_data[total_data['availability_365'] != 0]
print(total_data.shape)
total_data.head()

# Vuelvo a representar los histogramas de las variables categóricas para ver como han cambiado los datos con esta modificación
fig, axis = plt.subplots(4, 2, figsize = (18, 15), gridspec_kw={'height_ratios': [6, 1, 6, 1]})

# Crear una figura múltiple con histogramas y diagramas de caja
sns.histplot(ax = axis[0, 0], data = total_data, x = "price") # Hay un montón de datos atípicos
sns.boxplot(ax = axis[1, 0], data = total_data, x = "price")
sns.histplot(ax = axis[0, 1], data = total_data, x = "minimum_nights") # Hay un montón de datos atípicos
sns.boxplot(ax = axis[1, 1], data = total_data, x = "minimum_nights")
sns.histplot(ax = axis[2, 0], data = total_data, x = "availability_365")
sns.boxplot(ax = axis[3, 0], data = total_data, x = "availability_365")
sns.histplot(ax = axis[2, 1], data = total_data, x = "reviews_per_month")
sns.boxplot(ax = axis[3, 1], data = total_data, x = "reviews_per_month")


plt.tight_layout()

plt.show()

# 4. Análisis de variables multivariante
#   Numérico-numérico
fig, axis = plt.subplots(2, 3, figsize = (15, 7))

# Crear un diagrama de dispersión múltiple
sns.regplot(ax = axis[0, 0], data = total_data, x = "minimum_nights", y = "price")
sns.heatmap(total_data[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = total_data, x = "availability_365", y = "price").set(ylabel=None)
sns.heatmap(total_data[["price", "availability_365"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])
sns.regplot(ax = axis[0, 2], data = total_data, x = "reviews_per_month", y = "price").set(ylabel=None)
sns.heatmap(total_data[["price", "reviews_per_month"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 2])


# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

# Según las gráficas, el precio no guarda relación con las noches mínimas, con la disponibilidad ni con las reviews por mes

#   Categórico-categórico
fig, ax = plt.subplots(figsize=(10, 5))

# Crear el gráfico countplot
sns.countplot(ax=ax, data=total_data, x="neighbourhood_group", hue="room_type")


plt.tight_layout()

plt.show()

# Manhatan es el barrio que mas alquila inmuebles enteros

#   Análisis de correlaciones
total_data["neighbourhood_group"] = pd.factorize(total_data["neighbourhood_group"])[0]
total_data["room_type"] = pd.factorize(total_data["room_type"])[0]
total_data["host_name"] = pd.factorize(total_data["host_name"])[0]
total_data["neighbourhood"] = pd.factorize(total_data["neighbourhood"])[0]

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[["neighbourhood_group","room_type","neighbourhood","host_name"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

# Existe alta correlación entre el grupo de barrios y los barrios, lo cuál es lógico

#   Análisis numérico-categórico -> cálculo de correlaciones
fig, axis = plt.subplots(figsize = (10, 7))

sns.heatmap(total_data[["price","neighbourhood_group","availability_365","room_type","reviews_per_month","neighbourhood","host_name"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

#   Existe algo de correlación entre el precio y el tipo de habitaciones. El precio aumenta cuando las habitaciones son privadas o se alquila el piso entero.
#   price-room-type
#fig, axis = plt.subplots(figsize = (10, 5), ncols = 1)

sns.regplot(data = total_data, x = "price", y = "room_type")


plt.tight_layout()

plt.show()

total_data.describe()

# 5. Ingeniería de características
#   Análisis de outliers -> en las variables numéricas teníamos bastantes outliers (precio y noches mínimas)
#Precio -> me invento que AirBnb tiene una norma que limita el precio a 3000 dólares la noche
total_data = total_data[total_data['price'] < 500]
#Noches mínimas -> me invento que AirBnb tiene una norma que los usuarios pueden seleccionar como maximo 30 días minimos (1 mes)
total_data = total_data[total_data['minimum_nights'] < 30]
total_data.shape

# Vamos a ver cómo quedan ahora los histogramas con la eliminación de esos outliers
fig, axis = plt.subplots(4, 2, figsize = (18, 15), gridspec_kw={'height_ratios': [6, 1, 6, 1]})

# Crear una figura múltiple con histogramas y diagramas de caja
sns.histplot(ax = axis[0, 0], data = total_data, x = "price") # Hay un montón de datos atípicos
sns.boxplot(ax = axis[1, 0], data = total_data, x = "price")
sns.histplot(ax = axis[0, 1], data = total_data, x = "minimum_nights") # Hay un montón de datos atípicos
sns.boxplot(ax = axis[1, 1], data = total_data, x = "minimum_nights")
sns.histplot(ax = axis[2, 0], data = total_data, x = "availability_365")
sns.boxplot(ax = axis[3, 0], data = total_data, x = "availability_365")
sns.histplot(ax = axis[2, 1], data = total_data, x = "reviews_per_month")
sns.boxplot(ax = axis[3, 1], data = total_data, x = "reviews_per_month")


plt.tight_layout()

plt.show()

# Nuevas conclusiones -> los propietarios permite las estancias cortas

# Repito correlaciones por si han cambiado después de la limpieza de outliers
fig, axis = plt.subplots(figsize = (10, 7))

sns.heatmap(total_data[["price","neighbourhood_group","availability_365","room_type","reviews_per_month","neighbourhood","host_name"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

# El precio está relacionado con el tipo de habitación

#   Análisis de valores faltantes ->sólo faltan en las reviews y no son importantes para hacer un análisis del precio

total_data.isnull().sum().sort_values(ascending=False)

#   Escalado de valores
# División del conjunto en train y test
from sklearn.model_selection import train_test_split

# El precio se ha deducido que sólo guarda relación con el tipo de habitación y más débilmente con el barrio
num_variables = ["neighbourhood", "room_type"]

# Dividimos el conjunto de datos en muestras de train y test
X = total_data.drop("price", axis = 1)[num_variables]
y = total_data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_norm = scaler.transform(X_train)
X_train_norm = pd.DataFrame(X_train_norm, index = X_train.index, columns = num_variables)

X_test_norm = scaler.transform(X_test)
X_test_norm = pd.DataFrame(X_test_norm, index = X_test.index, columns = num_variables)

X_train_norm.head()

# 6. Selección de características -> no lo realizo porque sólo me quedaron dos características relevantes

X_train.to_csv("/workspaces/EDA/data/processed/clean_airbnb_train.csv", index=False)
X_test.to_csv("/workspaces/EDA/data/processed/clean_airbnb_test.csv", index=False)