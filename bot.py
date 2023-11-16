import cv2
import numpy as np
import tensorflow as tf

# Carregue o modelo
model = tf.keras.models.load_model('keras_model.h5')

# Inicialize a câmera
video = cv2.VideoCapture(0)

while True:
    # Verifique se a leitura do vídeo foi bem-sucedida
    check, frame = video.read()
    if not check:
        print("Erro ao ler o vídeo.")
        break

    # Redimensione a imagem
    img = cv2.resize(frame, (224, 224))

    # Converta a imagem em um array Numpy e aumente a dimensão
    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    # Normalize a imagem
    normalized_image = test_image / 255.0

    # Faça a previsão
    prediction = model.predict(normalized_image)

    # Exiba o resultado na imagem original (por exemplo, adicione texto)
    result_text = f"Previsão: {prediction}"
    cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exiba o resultado
    cv2.imshow("Resultado", frame)

    # Aguarde por uma tecla e verifique se é a tecla de espaço (32)
    key = cv2.waitKey(1)
    if key == 32:
        print("Fechando")
        break

# Libere o objeto de vídeo
video.release()
cv2.destroyAllWindows()
