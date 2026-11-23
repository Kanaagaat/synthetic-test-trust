# Glioma, T1 contrast-enhanced — prompt v4

Тот же промпт, что v3 (`glioma_t1c_v3.md`), плюс два изменения:

1. Строка про мягкость изображения в общей части (ниже почему).
2. **Число опухолей — как на референсе** (`{count_line}`). В тесте у 7 из 52 снимков с глиомой две опухоли, а 5 из 12 реальных пропусков бонус-модели — это как раз пропущенная вторая опухоль. Двойники с одной опухолью убрали бы этот тип ошибки из S-test и завысили бы recall.

Почему (мини-пилот v3, `viz/pilot_v3_stats.json`): генератор по умолчанию рисует резко. Резкость (дисперсия краёв) была близка к реальной (213–257 при 186 ± 56) только там, где вариант `{acquisition}` прямо просил мягкость или старый аппарат. Там, где про качество не было ни слова («head slightly off-center»), получилось 593–940. Теперь мягкость просим всегда, а `{acquisition}` отвечает только за разнообразие.

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
- Typical hospital image quality: slightly soft, moderate noise, like a JPEG exported from a
  clinical PACS viewer. Not a crisp, high-contrast textbook illustration.

Pathology:
- {count_line} of the same appearance type as the tumor(s) in the reference: the same
  kind of contrast enhancement, margins and internal structure as shown there.
- Give the tumor(s) a different shape and a different, anatomically plausible location
  than in the reference.
- Size of the largest tumor: about {size_pct}% of the image width.
- The tumor must be clearly visible to a radiologist. No other lesions.

Do not add: text, letters, numbers, scale bars, arrows, markers, bounding boxes, color,
multiple panels, borders, watermarks, or a 3D/illustration look.
```

## Вариации

| Параметр | Откуда |
|---|---|
| `{count_line}` | число рамок референса: «Exactly one glioma» / «Exactly N separate gliomas» |
| `{size_pct}` | ширина самой большой рамки референса в %, округлённая до целого |
| `{acquisition}` | случайно: older 1.5T scanner: lower resolution, visible noise, softer contrast; standard clinical scanner, slightly soft image, moderate noise; darker window, overall dim image; slightly brighter window, soft image; head slightly off-center and filling most of the frame; head centered with moderate black margins, slightly soft image |
