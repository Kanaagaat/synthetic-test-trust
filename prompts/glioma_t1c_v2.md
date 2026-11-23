# Glioma, T1 contrast-enhanced — prompt v2 (pilot v2)

Отличия от v1 (`glioma_t1c.md`):

- **Один референс на картинку** (mode `reference_guided`). Так можно измерить сходство каждой синтетики со своим референсом и проверить, не копирует ли генератор.
- Плоскость, размер и вид опухоли берутся из референса (`refs/pilot_v2/refs.json`), а расположение выбирается случайно. Задача генератора: «другой пациент с похожей опухолью».
- **Низкозлокачественная глиома переписана.** В v1 обе такие картинки вышли без видимой опухоли. Теперь просим явную гипоинтенсивную зону и масс-эффект.
- **Добавлен `{acquisition}`**: разный вид аппарата и экспорта. В v1 разброс яркости у синтетики был в 3–4 раза меньше, чем у реальных снимков.

## Prompt

```
Generate one new brain MRI slice. The attached image is a real reference from the same
dataset: match its MRI sequence and general look, but do not reproduce its anatomy, its
head shape or its tumor. This must be a different patient, with the tumor in a different
place.

Image:
- Plane: {plane}.
- T1-weighted post-contrast (gadolinium) MRI, single grayscale slice, black background,
  whole head visible.
- Acquisition and export: {acquisition}.

Pathology:
- Exactly one glioma, {location}.
- Size: {size}.
- Appearance: {pattern}.
- The tumor must be clearly visible to a radiologist. No other lesions.

Do not add: text, letters, numbers, scale bars, arrows, markers, bounding boxes, color,
multiple panels, borders, watermarks, or a 3D/illustration look.
```

## Вариации

| Параметр | Значения |
|---|---|
| `{location}` | случайно из списка v1 |
| `{size}` | small ≈8% / medium ≈13% / large ≈20% ширины снимка — по размеру опухоли на референсе |
| `{pattern}` | ring / heterogeneous / cystic — как в v1; **low_grade:** "non-enhancing or only faintly enhancing ill-defined mass that is clearly darker than the surrounding white matter, with visible mild mass effect such as effaced sulci or a slightly compressed ventricle" |
| `{acquisition}` | older 1.5T scanner: lower resolution, visible noise, softer contrast; modern 3T scanner: sharp, low noise; darker window, overall dim image; brighter window with higher contrast; head small in the frame with wide black margins; head slightly off-center and filling most of the frame |
