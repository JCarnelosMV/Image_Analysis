import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- TAMANHO DO PIXEL ---
pixel_size = 50 / 1024   # µm por pixel

# --- CARREGAR IMAGEM ---
image = cv2.imread("/content/Image_Analysis/imagens/imagem1.tif")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- INVERTER (pq poros são mais escuros) ---
inv = cv2.bitwise_not(gray)

# --- LIMIAR ADAPTATIVO ---
thresh = cv2.adaptiveThreshold(
    inv,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    35,   # tamanho da vizinhança (ajustável)
    -10   # constante (ajustável)
)

# --- LIMPEZA MORFOLÓGICA (abre os poros) ---
kernel = np.ones((3,3), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# --- CONTORNOS ---
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

areas_um2 = []
diameter_um = []

for cnt in contours:
    area_px = cv2.contourArea(cnt)
    if area_px < 20:   # remove ruído
        continue

    area_um2 = area_px * (pixel_size ** 2)
    areas_um2.append(area_um2)
    
    diam = 2 * np.sqrt(area_um2 / np.pi)
    diameter_um.append(diam)

print("\n--- RESULTADOS ---")
print(f"Poros detectados: {len(areas_um2)}")
print(f"Diâmetro médio (µm): {np.mean(diameter_um):.3f}")

# --- VISUALIZAÇÃO ---
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Segmentação")
plt.imshow(clean, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Contornos")
img_contours = image.copy()
cv2.drawContours(img_contours, contours, -1, (0,0,255), 1)
plt.imshow(img_contours)
plt.axis("off")

plt.show()

