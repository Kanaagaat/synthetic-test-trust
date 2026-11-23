# Glioma, T1 contrast-enhanced — prompt v3

Режим: один референс на картинку, референсы выбираются из пула (`refs/pools.json`), без ручных меток.

Что изменилось по сравнению с v2 и почему:

- **Тип опухоли берётся с референса, а не из наших категорий.** В пилоте v2 категория `low_grade` создала тёмные глиомы без накопления, которых почти нет в реальных данных. Бонус-модель пропустила 6 из 7 таких снимков. Теперь генератор повторяет тип опухоли референса, и распределение типов совпадает с реальным.
- **Плоскость — как у референса.** Метки плоскости не нужны.
- **Размер — из реальной рамки референса** (`{size_pct}`% ширины снимка; при нескольких рамках берётся самая большая).
- **Варианты `{acquisition}` смягчены.** В v2 «3T sharp» и «brighter window» дали дисперсию краёв 980–2275 при 186 ± 56 у реальных снимков, а «head small» — голову на 13–17% кадра при 47 ± 10%.

## Prompt

```
Generate one new brain MRI slice. The attached image is a real reference from the same
dataset: match its MRI sequence, its imaging plane and its general look, but do not
reproduce its anatomy, its head shape or its exact tumor. This must be a different patient.

Image:
- Same plane as the reference.
- T1-weighted post-contrast (gadolinium) MRI, single grayscale slice, black background,
  whole head visible.
- Acquisition and export: {acquisition}.

Pathology:
- Exactly one glioma of the same appearance type as the tumor in the reference: the same
  kind of contrast enhancement, margins and internal structure as shown there.
- Give it a different shape and put it in a different, anatomically plausible location
  than in the reference.
- Size: about {size_pct}% of the image width.
- The tumor must be clearly visible to a radiologist. No other lesions.

Do not add: text, letters, numbers, scale bars, arrows, markers, bounding boxes, color,
multiple panels, borders, watermarks, or a 3D/illustration look.
```

## Вариации

| Параметр | Откуда |
|---|---|
| `{size_pct}` | ширина самой большой рамки референса в %, округлённая до целого |
| `{acquisition}` | случайно: older 1.5T scanner: lower resolution, visible noise, softer contrast; standard clinical scanner, slightly soft image, moderate noise; darker window, overall dim image; slightly brighter window, soft image; head slightly off-center and filling most of the frame; head centered with moderate black margins, slightly soft image |
