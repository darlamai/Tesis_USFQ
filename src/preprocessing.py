"""
Preprocesamiento reproducible.  Se ejecuta aparte de los notebooks:

    python -m src.preprocesamiento          (desde la raíz del proyecto)

Contiene solo los pasos del notebook 01.eda: limpieza de filas con nulos, índice de fechas,
transformación a retornos / diferencias y partición en train / val / test.

Regla que organiza el módulo
----------------------------
* Estos pasos NO aprenden parámetros de los datos (cada retorno usa solo el precio de ese día y el
  anterior), así que se calculan sobre toda la serie y luego se parte por fechas. No hay fuga.
* Lo que sí aprende de los datos (diagnóstico de transformaciones, selección de variables, etc.)
  se decide solo con train y no está en este módulo.
* El test se guarda aparte y `cargar()` lo bloquea salvo que se pida explícitamente.
"""
import numpy as np
import pandas as pd

from . import config as cfg
from . import load_clean 


def construir_retornos(precios, transf=None):
    """
    Transformación punto a punto según config.TRANSF.
    Los precios <= 0 quedan en NaN en 'logret' (no se inventan valores).
    """
    transf = transf or cfg.transf
    out = pd.DataFrame(index=precios.index)
    for c, t in transf.items():
        s = precios[c]
        if t == "logret":
            out[c] = np.log(s.where(s > 0)).diff()
        elif t == "diff":
            out[c] = s.diff()
        else:
            raise ValueError(f"Transformación desconocida para {c}: {t}")
    return out.iloc[1:]          # el primer día no tiene retorno


def preparar_y_guardar(ruta_raw=None, destino=None):
    """
    crudo -> limpieza de filas con nulos -> retornos -> partición -> disco.
    Guarda precios_{train,val,test}.csv, retornos_{train,val,test}.csv y las filas eliminadas.
    """
    destino = destino or cfg.PROCESADOS
    destino.mkdir(parents=True, exist_ok=True)

    base, eliminadas = eliminar_filas_nulas(cargar_crudo(ruta_raw))
    ret = construir_retornos(base)

    px_partes, ret_partes = particionar(base), particionar(ret)
    verificar_particion(base, px_partes)
    verificar_particion(ret, ret_partes)

    for k in cfg.SPLITS:
        px_partes[k].to_csv(destino / f"precios_{k}.csv")
        ret_partes[k].to_csv(destino / f"retornos_{k}.csv")
    eliminadas.to_csv(destino / "filas_eliminadas_por_nulos.csv")

    return {k: len(v) for k, v in ret_partes.items()} | {"filas_eliminadas": len(eliminadas)}


def cargar(tipo, split, permitir_test=False, carpeta=None):
    """
    Lee un tramo ya procesado. tipo: 'retornos' | 'precios'. split: 'train' | 'val' | 'test'.
    El test está bloqueado: solo se abre en la evaluación final con permitir_test=True.
    """
    if split == "test" and not permitir_test:
        raise PermissionError(
            "El test está reservado para la evaluación final. "
            "Usa permitir_test=True solo en el notebook de evaluación."
        )
    carpeta = carpeta or cfg.PROCESADOS
    return pd.read_csv(carpeta / f"{tipo}_{split}.csv", index_col=0, parse_dates=True)


if __name__ == "__main__":
    print(preparar_y_guardar())