import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk


class HandGestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")

        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']
        self.labels_dict = {0: 'Hello', 1: 'Ok', 2: 'I Love You', 3: 'No'}

        self.label = tk.Label(root, text="Hand Gesture Recognition")
        self.label.pack()

        self.start_button = tk.Button(root, text="Start Camera", command=self.start_camera, bg="green")
        self.start_button.pack(pady=10)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit, bg="green")
        self.quit_button.pack(pady=10)

        self.phase_label = tk.Label(root, text="We use these labels for our project , we will provide more sign language words on these days", bg="gray")
        self.phase_label.pack()

        # Create a frame for the words
        self.words_frame = tk.Frame(root, bg="white")
        self.words_frame.pack(pady=10)

        # Place the words in the middle of the GUI
        for label_index, label_text in self.labels_dict.items():
            label = tk.Label(self.words_frame, text=label_text, bg="white")
            label.grid(row=0, column=label_index, padx=10)

        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        self.mp_hands.HAND_CONNECTIONS,  # hand connections
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                x_ = []
                y_ = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10
                x2 = int(max(x_) * frame.shape[1]) - 10
                y2 = int(max(y_) * frame.shape[0]) - 10

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


root = tk.Tk()
app = HandGestureRecognitionApp(root)
root.mainloop()
