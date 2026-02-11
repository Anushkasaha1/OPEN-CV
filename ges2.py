import cv2
import mediapipe as mp
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class LanguageCollector:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
        self.mp_hands=mp.solutions.hands
        self.hand=self.mp_hands.Hands()
        self.mp_draw=mp.solutions.drawing_utils
        
        self.label=[]
        self.data=[]
    def collect_data(self,sign_name,num_samples=1):
        """collect hand landmarks data for a specific sign"""
        print(f"collecting data for {sign_name}...")
        count=0
        started=False
        while count<num_samples:
            isTrue,frame=self.cap.read()
            if not isTrue:
                print("Failed to grab frame")
                break
            frame=cv2.flip(frame,1)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=self.hand.process(rgb)
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame,hand,self.mp_hands.HAND_CONNECTIONS)
                    if started:
                        landmarks=[]
                        for lm in hand.landmark:
                            landmarks.extend([lm.x,lm.y,lm.z])
                        self.data.append(landmarks)
                        self.label.append(sign_name)
                        count+=1
                        print(f"collected")
                    cv2.putText(frame,f"{sign_name}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    
            cv2.imshow("data",frame) 
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                started = True
            elif key == ord('q'):
                break
    def save(self,file_name="data.pkl"):
        """save collected data"""
        with open(file_name,"wb") as f:
            pickle.dump({'data':self.data,'label':self.label},f)
        print(f"data saved to {file_name}")   
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()



    def train_model(self,file_name="data.pkl",model_file="model.pkl"):
        """train a classifier"""
        with open(file_name,"rb") as f:
            dataset=pickle.load(f) 
        X=np.array(dataset['data'])
        Y=np.array(dataset['label'])   
        model=RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(X,Y)
        with open(model_file,"wb") as f:
            pickle.dump(model,f)    

        print(f"Model trained and saved to {model_file}")



class recognizer:
    def __init__(self,model_file="model.pkl"):
        self.model_file=model_file
        self.cap=cv2.VideoCapture(0)
        self.mp_hands=mp.solutions.hands
        self.hand=self.mp_hands.Hands()
        self.mp_draw=mp.solutions.drawing_utils
        with open(model_file,"rb") as f:
            self.model=pickle.load(f)
    def recognize(self):
        while True:
            isTrue,frame=self.cap.read()
            if not isTrue:
                print("Failed to grab frame")
                break
            frame=cv2.flip(frame,1)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=self.hand.process(rgb)
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame,hand,self.mp_hands.HAND_CONNECTIONS)
                    landmarks=[]
                    for lm in hand.landmark:
                        landmarks.extend([lm.x,lm.y,lm.z])
                    prediction=self.model.predict([landmarks])
                    cv2.putText(frame,f"sign:{prediction[0]}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    
            cv2.imshow("recognizer",frame) 
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand.close()


# USAGE:
# 1. Collect data for each sign
if __name__ == "__main__":
    # Collect data
    collector = LanguageCollector()
    
    # Collect samples for different signs
    collector.collect_data("thumbs_up", num_samples=1)
    collector.collect_data("thumbs_down", num_samples=1)
    collector.collect_data("swag", num_samples=1)
    collector.collect_data("cheese", num_samples=1)
    
    collector.save()
    collector.release()
    
    # Train model
    collector.train_model()
    
    # Use the recognizer
    recognizer = recognizer()
    recognizer.recognize()

        





