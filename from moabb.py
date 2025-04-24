from moabb.datasets import (
    Zhou2016,
    BNCI2014001,
    BNCI2014004,
    MunichMI,
    Weibo2014,
    Cho2017,
)

# Crear instancias de cada dataset para forzar su descarga
datasets = [
    Zhou2016(),
    BNCI2014001(),
    BNCI2014004(),
    MunichMI(),
    Weibo2014(),
    Cho2017(),
]

# Descargar los datos de cada sujeto
for ds in datasets:
    print(f"Descargando dataset: {ds.code}")
    subjects = ds.subject_list
    for subj in subjects:
        try:
            _ = ds.get_data(subjects=[subj])
        except Exception as e:
            print(f"Error al descargar sujeto {subj} del dataset {ds.code}: {e}")
