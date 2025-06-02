import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch, shuffle
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Новый размер
target_size = (128, 128)

def predict_new(image_path, model, pca):
    image = Image.open(image_path).convert('L')
    image_res = image.resize(target_size[::-1])
    image_array = np.array(image_res).flatten().reshape(1, -1)
    image_pca = pca.transform(image_array)

    prob = model.predict_proba(image_pca)[0]
    print(prob)
    max_prob = np.max(prob)
    predicted_label_num = np.argmax(prob)

    predicted_label = faces.target_names[predicted_label_num]

    plt.imshow(image_array.reshape(target_size), cmap='gray')
    plt.title(f"Предикт: {predicted_label}\n {max_prob:.2f}")
    plt.axis('off')
    plt.show()

    return predicted_label, max_prob

def load_faces(dataset_dir):
    images = []
    labels = []
    label_encoder = LabelEncoder()

    for label_folder in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_folder)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                image_path = os.path.join(label_path, filename)
                if image_path.endswith('.jpg') or image_path.endswith('.png'):
                    image = Image.open(image_path).convert('L')
                    image_resized = image.resize(target_size[::-1])
                    images.append(np.array(image_resized).flatten())
                    labels.append(label_folder)

    labels = label_encoder.fit_transform(labels)
    return Bunch(data=np.array(images), target=np.array(labels), target_names=label_encoder.classes_)

dataset_dir = 'faces_dataset'
faces = load_faces(dataset_dir)
print("Классы:", faces.target_names)
print("Размер данных:", faces.data.shape)

# Визуализация: 3 строки по 10 изображений
fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(14, 6))
for class_idx, person in enumerate(np.unique(faces.target)):
    person_images = faces.data[faces.target == person]
    for img_idx in range(min(10, len(person_images))):
        ax = axes[class_idx, img_idx]
        ax.imshow(person_images[img_idx].reshape(target_size), cmap='bone')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(faces.target_names[person], fontsize=8)
plt.tight_layout()
plt.show()

# Деление 80/20 по каждому классу
Xtrain, Xtest, ytrain, ytest = [], [], [], []
for label in np.unique(faces.target):
    class_indices = np.where(faces.target == label)[0]
    class_indices = shuffle(class_indices, random_state=42)
    split_idx = int(0.8 * len(class_indices))
    Xtrain.extend(faces.data[class_indices[:split_idx]])
    ytrain.extend(faces.target[class_indices[:split_idx]])
    Xtest.extend(faces.data[class_indices[split_idx:]])
    ytest.extend(faces.target[class_indices[split_idx:]])

Xtrain, ytrain = np.array(Xtrain), np.array(ytrain)
Xtest, ytest = np.array(Xtest), np.array(ytest)

# PCA и модель
n_components = min(100, Xtrain.shape[0])
pca = PCA(n_components=n_components, whiten=True, random_state=42)
pca.fit(Xtrain)
Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=2)
clf.fit(Xtrain_pca, ytrain)
yfit = clf.predict(Xtest_pca)

# Визуализация предсказаний
fig, ax = plt.subplots(1, len(Xtest), figsize=(12, 3))
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(target_size), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == ytest[i] else 'red')
plt.show()

# Метрики и матрица ошибок
print(classification_report(ytest, yfit, target_names=faces.target_names))
mat = confusion_matrix(ytest, yfit)
plt.figure(figsize=(5, 5))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title("Confusion Matrix")
plt.show()

# Проверка новых изображений
if os.path.exists('test'):
    for filename in os.listdir('test'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_p = os.path.join('test', filename)
            pred, prob = predict_new(image_p, clf, pca)
            print(f"Предикт: {pred} {prob:.2f}")
