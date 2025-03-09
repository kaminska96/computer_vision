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
    img = Image.open(img_path)
    # підвищення контрасту
    img_contrast = img.point(lambda p: p * 1.5)
    img_contrast.save(output_path)

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



image_path = "pic2.jpg"
output_contrast_image_path = "contrast_pic2.jpg"
output_sharpened_image_path = "sharpened_pic2.jpg"
show_image(image_path)
get_image_info(image_path)
show_brightness_histogram(image_path)
improve_contrast(image_path, output_contrast_image_path)
apply_filters(image_path)
sharpen_image(image_path, output_sharpened_image_path)

