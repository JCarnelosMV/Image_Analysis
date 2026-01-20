import cv2
import numpy as np
from skimage import filters, morphology, measure
import os

def analyze_pores(image_path, pixel_size_um=None):
    print(f"Analisando: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("⚠️ Erro ao abrir a imagem.")
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    thresh_val = filters.threshold_otsu(blurred)
    binary = blurred < thresh_val

    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=50)

    labels = measure.label(binary)
    props = measure.regionprops(labels)

    pore_areas = [p.area for p in props]

    porosity = np.sum(binary) / binary.size

    if pixel_size_um:
        pore_areas_um2 = [a * (pixel_size_um**2) for a in pore_areas]
        mean_pore_area = np.mean(pore_areas_um2)
    else:
        mean_pore_area = np.mean(pore_areas)

    return {
        "porosity_fraction": porosity,
        "mean_pore_area": mean_pore_area,
        "n_pores": len(pore_areas)
    }


if __name__ == "__main__":
    images_folder = "images"

    if not os.path.exists(images_folder):
        print("⚠️ Pasta 'images/' não encontrada. Crie e coloque as imagens MEV.")
        exit()

    images = [f for f in os.listdir(images_folder) if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]

    if not images:
        print("⚠️ Nenhuma imagem encontrada na pasta 'images/'.")
        exit()

    for img_file in images:
        result = analyze_pores(os.path.join(images_folder, img_file), pixel_size_um=None)

        if result:
            print(f"\n📌 Resultados para {img_file}:")
            print(f" - Porosidade: {result['porosity_fraction']:.3f}")
            print(f" - Tamanho médio dos poros: {result['mean_pore_area']:.2f}")
            print(f" - Número de poros detectados: {result['n_pores']}")
