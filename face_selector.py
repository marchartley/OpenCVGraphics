import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        param['center_x'] = x
        param['center_y'] = y

circle_params = {
    'center_x': 0,
    'center_y': 0
}

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback, circle_params)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

ret, frame = cap.read()
sprit = frame.copy()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    if not ret:
        print("Erreur : Impossible de recevoir le flux de la caméra.")
        break
    
    radius = abs(min(int(h/2) - circle_params['center_y'], int(h/2)))

    cv2.circle(frame, (int(w/2), int(h/2)), radius, (0, 255, 0), 2)
    cv2.putText(frame, "Press 's' to save.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        center_x, center_y = int(w/2), int(h/2)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        frame_with_alpha = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2BGRA)
        frame_with_alpha[:, :, 3] = mask

        cv2.imwrite('output.png', frame_with_alpha)
        break

cap.release()
cv2.destroyAllWindows()
