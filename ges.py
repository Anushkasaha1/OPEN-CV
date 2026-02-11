import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def fingers_count(hand):
    """count extended fingers"""
    finger_tips=[4,8,12,16,20]
    finger_joints=[2,6,10,14,18]
    fingers_up=0

    if hand.landmark[finger_tips[0]].x<hand.landmark[finger_joints[0]].x:

        fingers_up+=1
    for i in range(1,5):
        if hand.landmark[finger_tips[i]].y<hand.landmark[finger_joints[i]].y:
            fingers_up+=1
    return fingers_up

signs={
    0:"none",
    1:"one",
    2:"two/cheese/swag",
    3:"three",
    4:"four",
    5:"five"
}
while True:
    isTrue, frame = cap.read()
    if not isTrue:
        print("Failed to grab frame")
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # if hands detected
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            count=fingers_count(hand)
            get_sign=signs[count]
            cv2.putText(frame,f"sign:{get_sign}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.imshow("gesvid", frame)
    
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()