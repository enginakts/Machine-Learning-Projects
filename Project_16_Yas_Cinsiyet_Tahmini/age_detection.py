import cv2
import numpy as np

# Webcam'den görüntü almak için VideoCapture nesnesi oluşturuluyor.
cap = cv2.VideoCapture(0)
cap.set(3, 720)  # Video genişliği
cap.set(4, 1280)  # Video yüksekliği

# Yaş ve cinsiyet tahmini için model ortalama değerleri ve kategoriler belirleniyor.
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['ERKEK','KADIN' ]

# Caffe modellerini yükleyen fonksiyon
def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')  # Daha iyi yüz algılama modeli
    return age_net, gender_net, face_net

# Yüz algılama ve yaş/cinsiyet tespiti yapan fonksiyon
def detect_age_gender(image, age_net, gender_net, face_net):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)
    
    # Yüz tespiti
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Algılama doğruluğunu artır (önceden 0.5 idi)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x1, y1 = box.astype("int")

            # Yüz bölgesi kırpılıyor ve işleniyor
            face_img = image[y:y1, x:x1].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)

            # Cinsiyet tahmini
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Yaş tahmini
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Sonuçları ekrana yazdır
            overlay_text = f"{gender} {age}"
            print(f"Cinsiyet: {gender}, Yaş Aralığı: {age}")

            # Yüzü dikdörtgen içine al
            cv2.rectangle(image, (x, y), (x1, y1), (255, 255, 0), 2)

            # Tahmini bilgileri ekrana yazdır
            cv2.putText(image, overlay_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return image

if __name__ == "__main__":
    try:
        age_net, gender_net, face_net = load_caffe_models()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı.")
                break
            
            # Yaş ve cinsiyet tespiti yap
            result_frame = detect_age_gender(frame, age_net, gender_net, face_net)
            
            try:
                # Sonuçları göster
                cv2.imshow('Age and Gender Detection', result_frame)
                
                # 'q' tuşuna basıldığında çıkış yap
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error as e:
                print("OpenCV pencere hatası:", e)
                break
            
    except Exception as e:
        print("Bir hata oluştu:", e)
    finally:
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()
