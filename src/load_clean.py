import pandas as pd

from . import config as cfg

def cargar_crudo(ruta=None):
    """Lee el CSV crudo y devuelve precios con Date como DatetimeIndex ordenado."""
    ruta = ruta or cfg.raw_csv
    base = pd.read_csv(ruta)
    base["Date"] = pd.to_datetime(base["Date"])
    base = base.set_index("Date").sort_index()
    return base[cfg.columnas]


def resumen_nulos(base):
    """Cantidad y porcentaje de nulos por columna."""
    return pd.DataFrame({
        "cantidad": base.isna().sum(),
        "porcentaje": (base.isna().mean() * 100).round(2),
    }).sort_values("cantidad", ascending=False)


def eliminar_filas_nulas(base, umbral=None):
    """
    Elimina las filas con algún nulo si el porcentaje medio de nulos es <= umbral.
    Devuelve (base_limpia, filas_eliminadas) para dejar trazabilidad de lo que se quitó.
    """
    umbral = cfg.umbral_nulos if umbral is None else umbral
    filas = base[base.isna().any(axis=1)]
    if base.isna().mean().mean() <= umbral:
        return base.dropna(), filas
    print("El porcentaje de nulos supera el umbral: no se eliminan filas, revisar a mano.")
    return base, filas.iloc[0:0]

def particionar(df, splits=None):
    """Parte un DataFrame con índice de fechas en train / val / test según config.SPLITS."""
    splits = splits or cfg.SPLITS
    return {k: df.loc[a:b] for k, (a, b) in splits.items()}


def verificar_particion(df, partes, ventanas=None):
    """Sin solapes, sin huecos, y las ventanas de test cubren exactamente el test."""
    ventanas = ventanas or cfg.VENTANAS_TEST
    assert partes["train"].index.max() < partes["val"].index.min()
    assert partes["val"].index.max() < partes["test"].index.min()
    a0, b0 = cfg.SPLITS["train"][0], cfg.SPLITS["test"][1]
    assert sum(len(v) for v in partes.values()) == len(df.loc[a0:b0])
    assert sum(len(df.loc[a:b]) for a, b in ventanas.values()) == len(partes["test"])
    return True