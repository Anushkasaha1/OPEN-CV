import cv2
import mediapipe as mp
import numpy

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

points = []  # stores all drawn points
prev_x, prev_y = 0, 0

while True:
    isTrue, frame = cap.read()
    if not isTrue:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            index_tip = handLms.landmark[8]
            curr_x = int(index_tip.x * frame.shape[1])
            curr_y = int(index_tip.y * frame.shape[0])

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = curr_x, curr_y

            # Save the point pair
            points.append((prev_x, prev_y, curr_x, curr_y))

            prev_x, prev_y = curr_x, curr_y
    else:
        prev_x, prev_y = 0, 0  # reset when hand leaves

    # Redraw ALL saved points on every fresh frame
    for p in points:
        cv2.line(frame, (p[0], p[1]), (p[2], p[3]), (0, 255, 0), 5)

    cv2.imshow("Draw", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        break
    elif key == ord('c'):
        points = []  # clear all drawings

cap.release()
cv2.destroyAllWindows()
hand.close()