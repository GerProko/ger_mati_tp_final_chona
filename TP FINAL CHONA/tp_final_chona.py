import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf

modelo = tf.keras.models.load_model('ia_perros_y_gatos.h5')
image = Image.open('perro.jpg') # agregar la imagen a la carpeta y poner el nombre aca
img_resize = image.resize((100, 100)).convert('L')
np_img = np.array(img_resize)
prediccion = modelo.predict(np.expand_dims(np_img, 0))*100
if(prediccion >= 50):
    prediccion = 'Perro'
else:
    prediccion = 'Gato'
plt.imshow(img_resize)
plt.title('El animal de la imagen es un: ' + prediccion)
plt.show()

