import numpy as np
from PIL import Image
import os
from keras import layers
import matplotlib.pyplot as plt
import matplotlib.colors as col


class HOPEFIELD_NETWORK:
    def __init__(self, dimensiones_imagen: tuple = (64,64)) -> None:
        self.__dimensiones_imagen = dimensiones_imagen
        self.__datos = None
        self.__path = os.getcwd()
        self.__W = None
        self.__v0 = None


    def get_images(self, ruta_hacia_imagenes, train:bool = True, val:bool=False):
        ruta = os.path.join(self.__path, ruta_hacia_imagenes)
        self.__datos= np.zeros(shape=(self.__dimensiones_imagen[0]*self.__dimensiones_imagen[1], len(os.listdir(ruta))))

        for i, img in enumerate(os.listdir(ruta)):
            imagen_ruta = os.path.join(ruta, img)

            # Cargar la imagen
            imagen = Image.open(imagen_ruta)
            imagen = imagen.resize(self.__dimensiones_imagen)
            # Convertir la imagen a escala de grises
            imagen_gris = imagen.getchannel("A")

            # Convertir la imagen a un arreglo de numpy
            arreglo = np.array(imagen_gris)

            # Normalizar los valores de píxeles entre 0 y 1
            arreglo_normalizado = arreglo / 255.0

            # Convertir los valores de píxeles a 0 y 1 (umbral de 0.5)
            arreglo_binario = (arreglo_normalizado > 0.5).astype(np.uint8)

            aplanado = np.reshape(arreglo_binario,self.__dimensiones_imagen[0]*self.__dimensiones_imagen[1])
            
            f_vect = np.vectorize(self.__funcion_trans)

            aplanado = f_vect(aplanado) 
            
            self.__datos[:, i] = aplanado

    def fit(self):
        W_inicial = np.dot(self.__datos, np.transpose(self.__datos))

        self.__W = W_inicial - self.__datos.shape[1]*np.identity(self.__dimensiones_imagen[0]*self.__dimensiones_imagen[1])
        self.__W = self.__W * (1/self.__datos.shape[1]*self.__datos.shape[0])

    def activation(self, prevector, vector):
        for i in range(len(vector)):
            if vector[i] == 0:
                vector[i] = prevector[i]

            elif vector[i] > 0 :
                vector[i] = 1

            else:
                vector[i] = -1

        return vector
    
    def __view_image(self, vector):
        matriz = np.reshape(vector, newshape=(self.__dimensiones_imagen[0],self.__dimensiones_imagen[1]))
        # Mostrar la matriz como una imagen
        colores = ['blue', 'red']
        cmap_personalizado =  col.ListedColormap(colores)
        plt.imshow(matriz, cmap=cmap_personalizado)  # cmap='gray' para visualizar en escala de grises

        # Configuraciones opcionales
        plt.axis('off')  # No mostrar los ejes x e y
        plt.title('Imagen de la matriz')  # Título de la imagen

        # Mostrar la imagen
        plt.show()

    def predict(self, memoria):
        
        i = 0
        energias = []
        while True:
            self.__view_image(memoria)
            primera_prediccio = np.dot(memoria, self.__W)
            energia = self.funcio_energia(memoria, primera_prediccio)
            primera_prediccio = self.activation(memoria, primera_prediccio)
            

            if energia in energias:
                return True
            else:
                memoria = primera_prediccio

            i += 1
            print(energia)
            energias.append(energia)

    def funcio_energia(self, memoria, predict):
        return np.dot(predict, np.transpose(memoria)) * - 1

    def __funcion_trans(self, valor):
        if valor == 0:
            return -1
        
        return 1
    def __funcion_trans_2(self, valor):
        if valor < 0:
            return -1
        
        return 1
    
    def __validation(self, ruta_hacia_imagenes):
        ruta = os.path.join(self.__path, ruta_hacia_imagenes)

        for i, img in enumerate(os.listdir(ruta)):
            imagen_ruta = os.path.join(ruta, img)

            # Cargar la imagen
            imagen = Image.open(imagen_ruta)
            imagen = imagen.resize(self.__dimensiones_imagen)

            # Convertir la imagen a escala de grises
            imagen_gris = imagen.getchannel("A")


            # Convertir la imagen a un arreglo de numpy
            arreglo = np.array(imagen_gris)

            # Normalizar los valores de píxeles entre 0 y 1
            arreglo_normalizado = arreglo / 255.0

            # Convertir los valores de píxeles a 0 y 1 (umbral de 0.5)
            arreglo_binario = (arreglo_normalizado > 0.5).astype(np.uint8)

            aplanado = np.reshape(arreglo_binario,self.__dimensiones_imagen[0]*self.__dimensiones_imagen[1])

            

            # Añadir el ruido al vector original
            aplanado = self.agregar_ruido(aplanado, 0.3)

            f_vect = np.vectorize(self.__funcion_trans)

            aplanado = f_vect(aplanado) 

            self.__v0 = aplanado

    def agregar_ruido(self, imagen, probabilidad):
        # Copia de la imagen original
        imagen_con_ruido = np.copy(imagen)
        
        # Generar una matriz de valores aleatorios entre 0 y 1 del mismo tamaño que la imagen
        aleatorios = np.random.random(imagen.shape)
        
        # Aplicar ruido a la imagen basado en la probabilidad dada
        imagen_con_ruido[aleatorios < probabilidad] = 1 - imagen_con_ruido[aleatorios < probabilidad]

        return imagen_con_ruido
    

    def train_predict(self):
        self.get_images("imagenes")
        self.__validation("val")
        self.fit()
        self.predict(self.__v0)
            
obj = HOPEFIELD_NETWORK(dimensiones_imagen=(128,128))
obj.train_predict()
