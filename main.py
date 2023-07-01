import cv2
import os
import operator
import keyboard

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk
import enchant

from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application :

class Application:

    def __init__(self):

        self.hs = enchant.Dict('en_US')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("model.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("model.h5")

        self.ct = {}
        self.ct['blank'] = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.sentence = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()
        
        keyboard.on_release_key("space", self.on_space)
        keyboard.on_release_key("enter", self.on_enter)
        keyboard.on_release_key("backspace", self.on_backspace)
        

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1: y2, x1: x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image=self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))

            self.panel4.config(text=self.word, font=("Courier", 30))

            self.panel5.config(text=self.sentence, font=("Courier", 30))

        self.root.after(5, self.video_loop)

    def predict(self, test_image):

        test_image = cv2.resize(test_image, (128, 128))

        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {}

        prediction['blank'] = result[0][0]

        inde = 1

        for i in ascii_uppercase:
            prediction[i] = result[0][inde]

            inde += 1

        # LAYER 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

        self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':

            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if (self.ct[self.current_symbol] > 60):

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.ct['blank'] = 0

                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0
            
            
    def on_space(self, key): 
        if len(self.word) > 0:
            self.sentence += self.word + " "
            self.word = ""
            
    def on_enter(self, key): 
        if (len(self.sentence) > 16):
                self.sentence = ""

        self.word += self.current_symbol
            
    def on_backspace(self, key): 
        if (len(self.word) > 0):
                self.word = self.word[:-1]

    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
