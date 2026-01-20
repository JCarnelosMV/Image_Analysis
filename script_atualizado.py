import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology

# -------------------------------------
# CONFIGURAÇÕES DA IMAGEM
# -------------------------------------
# Nome da imagem (PNG, JPG ou TIFF)
image_path = "imagem 1.png"

# Tamanho do pixel (µm/pixel) — ajuste conforme sua escala do MEV
pixel_size = 50/ 1024

# -------------------------------------
# CARREGAR IMAGEM
# -------------------------------------
image = io.imread(image_path)

# Converter pra grayscale se vier RGB
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

# -------------------------------------
# LIMIARIZAÇÃO (OTSU)
# -------------------------------------
thresh = filters.threshold_otsu(gray)
binary = gray < thresh      # poros aparecem como preto → inverter se necessário

# Remove objetos minúsculos (ruído)
binary = morphology.remove_small_objects(binary, min_size=20)

# -------------------------------------
# MEDIDAS DOS POROS
# -------------------------------------
labels = measure.label(binary)
props = measure.regionprops(labels)

diametros = []
for region in props:
    if region.area >= 20:
        # converter área em µm²
        area_um = region.area * (pixel_size ** 2)
        # diâmetro equivalente
        d_eq = 2 * np.sqrt(area_um / np.pi)
        diametros.append(d_eq)

diametro_medio = np.mean(diametros) if len(diametros) > 0 else 0

# -------------------------------------
# CALCULAR POROSIDADE
# -------------------------------------
area_total = binary.size
area_poros = binary.sum()
porosidade = (area_poros / area_total) * 100

# -------------------------------------
# RESULTADOS
# -------------------------------------
print("\n--- RESULTADOS ---")
print(f"Poros detectados: {len(props)}")
print(f"Diâmetro médio (µm): {diametro_medio:.3f}")
print(f"Porosidade (%): {porosidade:.2f}")

# -------------------------------------
# PLOTAGEM
# -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].imshow(gray, cmap='gray')
ax[0].set_title("Imagem Original")
ax[0].axis("off")

ax[1].imshow(binary, cmap='gray')
ax[1].set_title("Máscara dos Poros")
ax[1].axis("off")

plt.show()
