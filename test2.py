import cv2
from ultralytics import YOLO

model = YOLO('C:/Users/Mi/Document/pythonProject2/model-2.pt')

class_names = model.names

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Не удалось подключиться к камере ")
        break
    # Выполняем предсказание объектов на изображении
    results = model.predict(img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            class_name = class_names[class_id]
            label = f'{class_name}'

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow('Алфавит', img)

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()