import face_recognition
from PIL import Image
import os
import numpy as np
import emotionClassification

# Обнаруживание лиц на изображениях
def findFaces(foldername):
    files = os.listdir('./'+foldername)
    faces = []
    for file in files:
        # Загрузка файла изображения
        ext = ['.jpg','.JPG','.jpeg','.JPEG']
        if file.endswith(tuple(ext)):
            image = face_recognition.load_image_file(foldername+'/'+file)
            # Поиск лиц на фото
            face_locations = face_recognition.face_locations(image)
            # Отображение каждого из найденных лиц
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                
                if (pil_image.width!=pil_image.height):
                    if (pil_image.width>pil_image.height):
                        new_right = pil_image.width-(pil_image.width-pil_image.height)
                        bottom = pil_image.height
                        pil_image = pil_image.crop((0, 0, new_right, bottom))
                    else:
                        new_bottom = pil_image.height-(pil_image.height-pil_image.width)
                        right = pil_image.width
                        pil_image = pil_image.crop((0,0,right,new_bottom))
                currentSize = pil_image.size
                targetSize = (48, 48)
                # Обрабатываем лица, размер области которых больше или равен 48х48
                if (currentSize >= targetSize):
                    if (currentSize > targetSize):
                        # Преобразование размера в 48х48
                        pil_image.thumbnail(targetSize)
                    # Преобразование в черно-белый формат
                    pil_image = pil_image.convert('L')
                    img = np.array(pil_image)
                    img = np.reshape(img, (48, 48, 1))
                    faces.append(img)
    print('Обнаружено всего лиц:    ',len(faces))
    faces = np.array(faces)
    return faces

# Вывод на экран частотной характеристики эмоций
def emotionPercentage(emotions):
    print('\nЧастота классифицированных эмоций: ')
    for i in range(len(emotions)):
        print(emotionClassification.labels[i],':    ',round((emotions[i]/sum(emotions)*100), 2),'%')

# Запуск модуля для классификации эмоций
def imgModule(foldername):
    faces = findFaces(foldername)
    emotions = emotionClassification.predict_emotion(faces)
    emotionPercentage(emotions)
