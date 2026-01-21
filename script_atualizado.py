import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1) CONFIGURAÇÕES
# ======================================================

pixel_size = 100 / 1024   # µm por pixel (ajuste conforme sua imagem)
min_area_px = 800        # área mínima para considerar um poro (em pixels)
min_circularity = 0.20   # descarta objetos muito alongados

# ======================================================
# 2) CARREGAR IMAGEM
# ======================================================

image = cv2.imread("/content/Image_Analysis/images/fig1.tif")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# suaviza antes de threshold
gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

# ======================================================
# 3) SEGMENTAÇÃO (OTSU)
# ======================================================

# Otsu automaticamente acha o melhor limiar
_, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ======================================================
# 4) MORFOLOGIA PARA LIMPAR RUÍDO & UNIR PAREDES
# ======================================================

kernel = np.ones((2,2), np.uint8)

# fecha paredes quebradas
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# remove ruído fino
clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

# ======================================================
# 5) CONTORNOS
# ======================================================

contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

areas_um2 = []
diameters_um = []
valid_contours = []

for cnt in contours:
    
    area = cv2.contourArea(cnt)
    if area < min_area_px:
        continue
    
    per = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * (area / (per*per + 1e-6))

    # remover paredes finas (circularidade baixa)
    if circularity < min_circularity:
        continue

    # converte para µm²
    area_um2 = area * (pixel_size ** 2)
    areas_um2.append(area_um2)

    # diâmetro equivalente circular
    diam = 2 * np.sqrt(area_um2 / np.pi)
    diameters_um.append(diam)

    valid_contours.append(cnt)

# ======================================================
# 6) POROSIDADE
# ======================================================

total_pixels = clean.size
poro_pixels = clean.sum() / 255  # porque poro = 255
porosity_percent = (poro_pixels / total_pixels) * 100

# ======================================================
# 7) RESULTADOS
# ======================================================

print("\n--- RESULTADOS ---")
print(f"Poros detectados: {len(valid_contours)}")
print(f"Diâmetro médio (µm): {np.mean(diameters_um):.3f}")
print(f"Área média dos poros (µm²): {np.mean(areas_um2):.3f}")
print(f"Porosidade (%): {porosity_percent:.2f}")
print("-----------------------------------------")


# ======================================================
# 8) VISUALIZAÇÃO
# ======================================================

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original (gray)")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Segmentação refinada")
plt.imshow(clean, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Contornos válidos")
img_c = image.copy()
cv2.drawContours(img_c, valid_contours, -1, (0,0,255), 1)
plt.imshow(img_c)
plt.axis("off")

plt.show()

# ======================================================
# 9) HISTOGRAMA DOS DIÂMETROS
# ======================================================

plt.figure(figsize=(6,4))
plt.hist(diameters_um, bins=30)
plt.xlabel("Diâmetro dos poros (µm)")
plt.ylabel("Frequência")
plt.title("Distribuição dos diâmetros dos poros")
plt.show()

