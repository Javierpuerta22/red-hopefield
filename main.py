import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.rk import RegressionKriging
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd



dd = pd.read_csv("datos.csv")



# Crear una instancia de OrdinaryKriging y ajustar el modelo
OK = OrdinaryKriging(dd["longitud"], dd["longitud"], dd["temperatura"], variogram_model='power', coordinates_type="geographic", enable_plotting=True, enable_statistics=True)


# Realizar la interpolación por kriging en nuevos puntos
gridx = np.linspace(-120, 10, 100)
gridy = np.linspace(-120 , 10, 100)
z_interp, sigmasq = OK.execute('grid', gridx, gridy)

# Graficar el resultado de la interpolación
plt.imshow(z_interp, origin='lower', extent=(0, 5, 0, 2))
plt.colorbar(label='Interpolated Value')
plt.scatter(dd["longitud"], dd["longitud"], dd["temperatura"], cmap='jet', label='Data Points')
plt.legend()
plt.show()

