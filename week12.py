import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
import os


def show_image(img_path):
    # Завантаження зображення за допомогою OpenCV
    img_cv2 = cv2.imread(img_path)
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Завантаження зображення за допомогою PIL
    img_pil = Image.open(img_path)

    # Відображення обох зображень
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
    img = Image.open(img_path).convert("L")  # Конвертуємо зображення в відтінки сірого
    histogram = img.histogram()

    plt.figure(figsize=(8, 6))
    plt.plot(histogram, color='black')
    plt.title("Гістограма яскравості")
    plt.xlabel("Яскравість")
    plt.ylabel("Кількість пікселів")
    plt.show()


def improve_contrast(img_path, output_path):
    img = Image.open(img_path)
    img_contrast = img.point(lambda p: p * 1.5)  # Підвищення контрасту
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

    # Біквадратний фільтр (Bilateral Filter)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    # Відображення результатів
    plt.figure(figsize=(10, 7))
    titles = ['Original Image', 'Gaussian Filter', 'Median Filter', 'Bilateral Filter']
    images = [img, gaussian, median, bilateral]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.show()


def sharpen_image(img_path, output_path):
    img = cv2.imread(img_path)

    # Фільтр для підвищення різкості
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharpened = cv2.filter2D(img, -1, kernel)

    # Зберегти результат
    cv2.imwrite(output_path, img_sharpened)

    # Відображення результату
    plt.imshow(cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB))
    plt.title("Підвищена різкість")
    plt.axis('off')
    plt.show()


def threshold_segmentation(img_path):
    img = cv2.imread(img_path, 0)  # Завантаження зображення у відтінках сірого

    # Порогова сегментація з фіксованим порогом
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Порогова сегментація з використанням методу Otsu
    _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Відображення результатів
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

    # Операції морфології для відокремлення переднього та заднього планів
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_img, kernel, iterations=3)

    # Відстань трансформації для переднього плану
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Маркування
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Маркуємо компоненти
    _, markers = cv2.connectedComponents(sure_fg)

    # Додаємо 1 до всіх маркерів, щоб бути впевненими, що фон - 1
    markers = markers + 1
    markers[unknown == 255] = 0

    # Застосування алгоритму Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # Відображення результату
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Watershed сегментація")
    plt.axis('off')
    plt.show()


def grabcut_segmentation(img_path):
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Прямокутник для ініціалізації (визначте об'єкт)
    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

    # Використання алгоритму GrabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Створення маски для сегментованого зображення
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_result = img * mask2[:, :, np.newaxis]

    # Відображення результату
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


def process_video_stream(video_path=None):
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    back_sub = cv2.createBackgroundSubtractorMOG2()

    ret, first_frame = cap.read()
    if not ret:
        print("Не вдалося отримати перший кадр")
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    hsv_mask = np.zeros_like(first_frame)
    hsv_mask[..., 1] = 255

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # аналіз рухомих об'єктів за допомогою Optical Flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # перетворюємо Optical Flow на кут і величину
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_mask[..., 0] = angle * 180 / np.pi / 2
        hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        optical_flow_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

        # Background Subtraction для виділення рухомих об'єктів
        fg_mask = back_sub.apply(frame)

        cv2.imshow("Оригінальне відео", frame)
        cv2.imshow("Optical Flow", optical_flow_rgb)
        cv2.imshow("Background Subtraction", fg_mask)

        prev_gray = gray

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def scale_image(image_path, scale_x, scale_y):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не вдалося завантажити зображення з {image_path}")
        return
    scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)

    cv2.imshow(f'Масштабоване зображення {scale_x}x{scale_y}', scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(image_path, angle, center=None, scale=1.0):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не вдалося завантажити зображення з {image_path}")
        return
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    cv2.imshow(f'Повернуте зображення на {angle} градусів', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perspective_transform(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не вдалося завантажити зображення з {image_path}")
        return
    h, w = image.shape[:2]

    pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50], [w - 50, h - 50]])

    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv2.warpPerspective(image, M, (w, h))

    cv2.imshow('Перспективна трансформація', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def erode_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Не вдалося завантажити зображення з {image_path}")
        return
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)

    cv2.imshow('Ерозія', eroded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dilate_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Не вдалося завантажити зображення з {image_path}")
        return
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    cv2.imshow('Дилатація', dilated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def opening_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Не вдалося завантажити зображення з {image_path}")
        return
    kernel = np.ones(kernel_size, np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Відкриття', opened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def closing_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Не вдалося завантажити зображення з {image_path}")
        return
    kernel = np.ones(kernel_size, np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Закриття', closed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_example_images(X, y, n_row=3, n_col=6, h=64, w=64, title=""):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(X[i].reshape((h, w)), cmap='gray')
        plt.title("Face" if y[i] == 1 else "Non-Face", size=10)
        plt.axis('off')
    plt.suptitle(title, size=16)
    plt.show()


def load_images(folder_path, target_size=(64, 64), label=None):
    X = []
    y = []

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        try:
            img = Image.open(image_path).convert('L').resize(target_size)
            X.append(np.asarray(img).flatten())
            y.append(label)
        except Exception as e:
            print(f"Не вдалося завантажити {image_path}: {e}")

    return np.array(X), np.array(y)


def face_detection_classification():

    faces_folder = "Dataset/faces_folder"
    faces, faces_labels = load_images(faces_folder, target_size=(64, 64), label=1)

    non_faces_folder = "Dataset/non_faces_folder"
    non_faces, non_faces_labels = load_images(non_faces_folder, target_size=(64, 64), label=0)

    X = np.vstack((faces, non_faces))
    y = np.concatenate((faces_labels, non_faces_labels))

    # Розділення на train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print(f"Розмір тренувального набору: {X_train.shape}")
    print(f"Розмір тестового набору: {X_test.shape}")

    plot_example_images(X_train, y_train, h=64, w=64, title="Приклади навчальних зображень")

    # Зменшення розмірності за допомогою PCA
    pca = PCA(n_components=100, whiten=True, random_state=42).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Навчання моделей
    models = {
        "SVM": SVC(kernel='rbf', C=10, gamma=0.001),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}
    for name, model in models.items():
        print(f"\n{name}")
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"Точність: {acc:.2f}")
        print(classification_report(y_test, y_pred, target_names=["Non-Face", "Face"]))

    # Графік результатів
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
    plt.title("Порівняння точності моделей", pad=20)
    plt.ylabel("Точність")
    plt.ylim(0, 1)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.show()


def create_cnn_model(input_shape=(64, 64, 1), num_classes=2):
    model = models.Sequential([
        # 1 згортковий шар
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # 2 згортковий шар
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 3 згортковий шар
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 4 згортковий шар
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return model


def train_cnn_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    # нормалізація
    X_train = X_train.astype('float32') / 255
    X_val = X_val.astype('float32') / 255

    # перетворення міток у one-hot encoding
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # аугментація даних (week 12)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    datagen.fit(X_train.reshape((-1, 64, 64, 1)))

    model = create_cnn_model()

    # навчання моделі
    history = model.fit(
        datagen.flow(X_train.reshape((-1, 64, 64, 1)), y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val.reshape((-1, 64, 64, 1)), y_val))

    return model, history


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def evaluate_model(model, X_test, y_test):
    X_test = X_test.astype('float32') / 255
    y_test = to_categorical(y_test)

    test_loss, test_acc = model.evaluate(X_test.reshape((-1, 64, 64, 1)), y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')


def cnn_face_detection():
    faces_folder = "Dataset/faces_folder"
    faces, faces_labels = load_images(faces_folder, target_size=(64, 64), label=1)

    non_faces_folder = "Dataset/non_faces_folder"
    non_faces, non_faces_labels = load_images(non_faces_folder, target_size=(64, 64), label=0)

    X = np.vstack((faces, non_faces))
    y = np.concatenate((faces_labels, non_faces_labels))

    # розділення на train/val/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      random_state=42)

    print(f"Розмір тренувального набору: {X_train.shape}")
    print(f"Розмір валідаційного набору: {X_val.shape}")
    print(f"Розмір тестового набору: {X_test.shape}")

    # навчання моделі
    model, history = train_cnn_model(X_train, y_train, X_val, y_val, epochs=30)

    plot_training_history(history)

    evaluate_model(model, X_test, y_test)

    return model


def detect_faces_mobilenet(img_path):
    model = MobileNet(weights='imagenet')

    # підготовка зображення
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # класифікація зображення
    preds = model.predict(x)

    # декодування результатів
    print('Результати класифікації MobileNet:')
    for (imagenet_id, label, score) in decode_predictions(preds, top=3)[0]:
        print(f"{label}: {score:.2f}")

    img = cv2.imread(img_path)
    cv2.imshow('MobileNet Classification', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_faces_resnet(img_path):
    model = ResNet50(weights='imagenet')

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # класифікація зображення
    preds = model.predict(x)

    print('Результати класифікації ResNet50:')
    for (imagenet_id, label, score) in decode_predictions(preds, top=3)[0]:
        print(f"{label}: {score:.2f}")

    img = cv2.imread(img_path)
    cv2.imshow('ResNet Classification', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

# 6
process_video_stream("video.mp4")

#7
scale_image(image_path, 1.5, 1.5)
rotate_image(image_path, 45)
perspective_transform(image_path)

#8
erode_image(image_path, kernel_size=(3, 3))
dilate_image(image_path, kernel_size=(3, 3))
opening_image(image_path, kernel_size=(3, 3))
closing_image(image_path, kernel_size=(3, 3))

# 9
face_detection_classification()

# 10, 12
cnn_face_detection()

# 11
detect_faces_mobilenet('pic2.jpg')
detect_faces_resnet('pic2.jpg')