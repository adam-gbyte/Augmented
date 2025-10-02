Simple image augmentation toolkit using Pillow + numpy.

Fitur:
- rotasi (degree)
- flip horizontal / vertical
- zoom in / zoom out (scale)
- translasi (geser)
- perubahan brightness / contrast
- tambah noise gaussian
- pipeline random untuk memperbanyak data
- batch processing dari folder

Usage examples:
```bash
python augment.py --input path/to/image.jpg --outdir dataset/augmented/ --n 20
```
```bash
python augment.py --indir dataset/images --outdir dataset/augmented --n-per-image 5
```