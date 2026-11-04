# Синтетические фантомы

Файлы в этой папке создаются генератором и не хранятся в git:

```
cd core
python -m aortaplan_core.phantoms            # все стандартные фантомы
python -m aortaplan_core.phantoms --list     # список
python -m aortaplan_core.phantoms --only torus_arc helix
```

Для каждого фантома пишутся `<имя>.nrrd` (объём, HU, int16, пространство LPS — как у DICOM)
и `<имя>.truth.json` (истинные значения в RAS, мм: центральная линия, нормали, полуоси слоёв,
аналитические диаметры, площади и длины).
