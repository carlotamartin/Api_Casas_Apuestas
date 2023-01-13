from main.map import ApuestaSchema
from main.repositories.repositorioapuesta import ApuestaRepositorio
from main.repositories.repositoriocuota import CuotaRepositorio
from abc import ABC

apuesta_schema = ApuestaSchema()
apuesta_repositorio = ApuestaRepositorio()
cuota_repositorio = CuotaRepositorio()

class ApuestaService:

    def agregar_apuesta(self, apuesta, local, visitante):
        cuota = cuota_repositorio.find_by_partido(apuesta)
        probabilidad = self.set_cuota(cuota, local, visitante)
        apuesta.ganancia = round(apuesta.monto * probabilidad, 2)
        return apuesta_repositorio.create(apuesta)

    def set_cuota(self, cuota, local, visitante):
        if local:
            cuota_local = CuotaLocal()
            probabilidad = cuota_local.calcular_cuota(cuota)
            return probabilidad
        if visitante:
            cuota_visitante = CuotaVisitante()
            probabilidad = cuota_visitante.calcular_cuota(cuota)
            return probabilidad
        cuota_empate = CuotaEmpate()
        probabilidad = cuota_empate.calcular_cuota(cuota)
        return probabilidad

    def obtener_apuesta_por_id(self, id):
        return apuesta_repositorio.find_one(id)

    def obtener_apuestas_ganadas(self):
        return apuesta_repositorio.find_wins()

    def obtener_apuestas(self):
        return apuesta_repositorio.find_all()

import pickle
import numpy as np
class CuotaStrategy(ABC):
    #En el constructor, descargamos los datos y los preprocesamos
    def __init__(self):
        self.scaler = pickle.load(open("scalers/pt.pkl", "rb"))
        self.model = pickle.load(open("models/lm.pkl", "rb"))

    #En el método calcular_cuota, recibimos los ids de los equipos y devolvemos las cuotas
    def calcular_cuota(self, local_id, visitante_id):

        data = [[local_id, visitante_id]]

        #Estandarizamos los datos
        data_scaled = self.scaler.transform(data)

        # Hacemos la exponencial para que nos devuelta el valor real de la cuota
        cuotas_pred = np.exp(self.model.predict(data_scaled))

        #Nos va a devolver un array de 3 elementos, que son las cuotas de local, empate y visitante
        return cuotas_pred



class CuotaLocal(CuotaStrategy):
    def calcular_cuota(self, cuota):
        probabilidad = cuota.cuota_local
        return probabilidad

class CuotaVisitante(CuotaStrategy):
    def calcular_cuota(self, cuota):
        probabilidad = cuota.cuota_visitante
        return probabilidad

class CuotaEmpate(CuotaStrategy):
    def calcular_cuota(self, cuota):
        probabilidad = cuota.cuota_empate
        return probabilidad