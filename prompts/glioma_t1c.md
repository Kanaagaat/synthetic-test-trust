# Glioma, T1 contrast-enhanced — generation prompt (v1, pilot)

Как пользоваться: Codex получает референсы из `refs/` через `-i` и текст из блока «Prompt». Подставь `{plane}`, `{location}`, `{size}`, `{pattern}` из таблицы вариаций, чтобы снимки не были одинаковыми.

## Prompt

```
Generate one new brain MRI slice that looks like a real clinical image from the same
dataset as the attached reference images. Use the references only for style: contrast,
noise, sharpness, intensity range and framing. Do not copy any reference; this must be a
different patient with different anatomy.

Image:
- Plane: {plane}.
- T1-weighted post-contrast (gadolinium) MRI, single grayscale slice, square, black
  background, the whole head visible and centered, skull and scalp visible.
- Realistic MRI noise and slight blur, like a JPEG export from a PACS viewer.

Pathology:
- Exactly one glioma, {location}.
- Size: {size}.
- Appearance: {pattern}.
- Mild mass effect on nearby structures is allowed. No other lesions.

Do not add: text, letters, numbers, scale bars, arrows, markers, bounding boxes, color,
multiple panels, borders, watermarks, or a 3D/illustration look.
```

## Вариации

| Параметр | Значения |
|---|---|
| `{plane}` | axial; sagittal; coronal |
| `{location}` | in the left frontal lobe white matter; in the right frontal lobe; in the left temporal lobe; in the right parietal lobe; in the left parietal-occipital region; deep near the right lateral ventricle |
| `{size}` | small, about 8% of the image width; medium, about 13% of the image width; large, about 20% of the image width |
| `{pattern}` | irregular thick ring of enhancement with a dark necrotic center and darker surrounding edema; heterogeneous patchy enhancement with blurred, infiltrative margins; cystic mass with a thin enhancing rim and a small enhancing nodule; faint minimal enhancement, mostly hypointense, poorly defined borders (low-grade look) |

Размеры взяты из реальных рамок глиомы в train: медиана ≈13% ширины снимка, p10 ≈7%, p90 ≈21%.

Частоты плоскостей и паттернов стоит подобрать под реальные данные после пилота.

## Пилот

Референсы (все из train):

- `refs/gl_Tr_gl_0964_dce226.jpg` — sagittal, кольцевое накопление
- `refs/gl_Tr_gl_0374_3a0d34.jpg` — axial, неоднородная
- `refs/gl_Tr_gl_0969_4583a3.jpg` — sagittal, кистозная
- `refs/gl_Tr_gl_0336_b43433.jpg` — axial, лобная

Каждый сгенерированный снимок: перевести в grayscale, привести к 512×512, сохранить в `generated/pilot/` и записать строку в `generated/manifest.csv`.
