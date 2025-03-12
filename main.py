import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def show_image(img_path):
    # OpenCV
    img_cv2 = cv2.imread(img_path)
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # PIL
    img_pil = Image.open(img_path)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_cv2_rgb)
    axs[0].set_title("Зображення з OpenCV")
    axs[0].axis('off')

    axs[1].imshow(img_pil)
    axs[1].set_title("Зображення з PIL")
    axs[1].axis('off')

    plt.show()


def get_image_info(img_path):
    img = Image.open(img_path)
    print(f"\nImage: {img_path}")
    print(f"Формат: {img.format}")
    print(f"Розмір: {img.size}")
    print(f"Режим: {img.mode}")


def show_brightness_histogram(img_path):
    img = Image.open(img_path).convert("L")
    histogram = img.histogram()

    plt.figure(figsize=(8, 6))
    plt.plot(histogram, color='black')
    plt.title("Гістограма яскравості")
    plt.xlabel("Яскравість")
    plt.ylabel("Кількість пікселів")

    plt.show()


def improve_contrast(img_path, output_path):
    img_cv2 = cv2.imread(img_path)
    img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # підвищення контрасту
    contrast = 1.5
    img_contrast = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, 0)

    cv2.imwrite(output_path, img_contrast)

    plt.imshow(img_contrast)
    plt.title("Покращена контрасність")
    plt.axis('off')

    plt.show()


def apply_filters(img_path):
    img = cv2.imread(img_path, 0)

    # Гаусовий фільтр
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # Медіанний фільтр
    median = cv2.medianBlur(img, 5)

    # Біквадратний фільтр
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    plt.figure(figsize=(10, 7))
    titles = ['Original Image', 'Гаусовий фільтр', 'Медіанний фільтр', 'Біквадратний фільтр']
    images = [img, gaussian, median, bilateral]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.show()


def sharpen_image(img_path, output_path):
    img = cv2.imread(img_path)

    # фільтр для підвищення різкості
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharpened = cv2.filter2D(img, -1, kernel)

    cv2.imwrite(output_path, img_sharpened)

    plt.imshow(cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB))
    plt.title("Підвищена різкість")
    plt.axis('off')

    plt.show()


def threshold_segmentation(img_path):
    img = cv2.imread(img_path, 0)

    # порогова сегментація з фіксованим порогом
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # порогова сегментація з використанням методу Otsu
    _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(thresh_img, cmap='gray')
    plt.title("Фіксований поріг")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(otsu_img, cmap='gray')
    plt.title("Метод Otsu")
    plt.axis('off')

    plt.show()


def watershed_segmentation(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # відокремлення переднього та заднього планів
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_img, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # маркування
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Watershed сегментація")
    plt.axis('off')

    plt.show()


def grabcut_segmentation(img_path):
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

    # GrabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # маска для сегментованого зображення
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_result = img * mask2[:, :, np.newaxis]

    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title("GrabCut сегментація")
    plt.axis('off')

    plt.show()


def detect_edges(img_path):
    img = cv2.imread(img_path, 0)
    # алгоритм Canny
    edges = cv2.Canny(img, 100, 200)

    plt.imshow(edges, cmap='gray')
    plt.title("Контури облич (Canny)")
    plt.axis('off')

    plt.show()


def sift_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT детектор
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    img_sift = cv2.drawKeypoints(img, keypoints, None)

    plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Детектор")
    plt.axis('off')

    plt.show()


def surf_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(400)
    keypoints, descriptors = surf.detectAndCompute(gray, None)

    img_surf = cv2.drawKeypoints(img, keypoints, None)

    plt.imshow(cv2.cvtColor(img_surf, cv2.COLOR_BGR2RGB))
    plt.title("SURF Детектор")
    plt.axis('off')

    plt.show()


def orb_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB детектор
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    img_orb = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
    plt.title("ORB Детектор")
    plt.axis('off')

    plt.show()


def hog_descriptor(img_path):
    img = cv2.imread(img_path, 0)
    hog = cv2.HOGDescriptor()

    h = hog.compute(img)

    plt.plot(h)
    plt.title("HOG Дескриптори")

    plt.show()


image_path = "pic2.jpg"
output_contrast_image_path = "contrast_pic2.jpg"
output_sharpened_image_path = "sharpened_pic2.jpg"

# 1
show_image(image_path)
get_image_info(image_path)
# 2
show_brightness_histogram(image_path)
improve_contrast(image_path, output_contrast_image_path)
# 3
apply_filters(image_path)
sharpen_image(image_path, output_sharpened_image_path)
# 4
threshold_segmentation(image_path)
watershed_segmentation(image_path)
grabcut_segmentation(image_path)
# 5
detect_edges(image_path)
sift_detector(image_path)
orb_detector(image_path)
hog_descriptor(image_path)
