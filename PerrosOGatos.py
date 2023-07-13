# features 1=si y  0=no
# Criterios:
# Tiene el pelo largo?
# Tiene las uñas afiladas?
# Hace miau?

# A continuacion se le dara una base de datos

perro1 = [0, 1, 1]
perro2 = [1, 0, 1]
perro3 = [1, 1, 1]

gato1 = [0, 1, 0]
gato2 = [0, 1, 1]
gato3 = [1, 1, 0]

# Creamos nuestros datos
x_train = [perro1, perro2, perro3, gato1, gato2, gato3]
y_train = [1, 1, 1, 0, 0, 0]  # 1 si pertenece a perro y 0 si pertenece a gato

# Importamos sklearn
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(x_train, y_train)

animal_misterioso = [1, 1, 1]  # Animal que queremos predecir
print(model.predict([animal_misterioso]))  # Nos retorna que es un perro

# Otro ejemplo
misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

x_test = [misterio1, misterio2, misterio3]
y_test = [0, 1, 1]  # Clasificacion del ususario de animales misteriosos

# Le pedimos al modelo que haga una prediccion de animales misteriosos
print(model.predict(x_test))  # Nos retorna [1 0 1], el modelo acerto una de 3 veces


# Como obtener la métrica
predicciones = model.predict(x_test)
correctos = (predicciones == y_test).sum()
total = len(x_test)
tasa_de_acierto = correctos / total
print(f"La tasa de acierto fue de: {round(tasa_de_acierto*100,2)}%")


# La metrica más sencillo
from sklearn.metrics import accuracy_score

tasa_de_acierto = accuracy_score(y_test, predicciones)
print(f"La tasa de acierto fue de: {round(tasa_de_acierto*100,2)}%")
