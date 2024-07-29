import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import keras
from keras.preprocessing.image import img_to_array
import tensorflow as tf

class AgeingClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("SpectR.")

        self.root.geometry("600x400")   #Setting up the size of the GUI
        self.root.minsize(200,200)      #Setting the minimum size of GUI Window
        self.root.maxsize(300,700)      #Setting the maximum size of GUI Window
  

        self.label = tk.Label(root, text="Choose an option:")   #Here we have two options -> choose an existing image or take a new photo
        self.label.pack()

        self.choose_button = tk.Button(root, text="Choose Image from Folder", command=self.choose_image)
        self.choose_button.pack(pady=5)

        self.capture_button = tk.Button(root, text="Take a New Photo", command=self.capture_image)
        self.capture_button.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)   #Choosing an existing image
            self.predict(file_path)         #perfrom prediction on chosen image

    def capture_image(self):
        cap = cv2.VideoCapture(0)
       
        if not cap.isOpened():
          print("Error!! Could not open video stream from webcam")
        else:
          print("Press 'Enter' to capture an image or 'q' to quit")

        while True:
        
          ret, img = cap.read()   # Read an image from the webcam

          if not ret:
               print("Error!! Could not read image from webcam.")
               break

          cv2.imshow('Webcam View', img)   

          key = cv2.waitKey(1) & 0xFF    # Wait for the user to press a key and capture the image

          # Check if 'Enter' key is pressed to capture the image
          if key == 13:  # 13 -> the Enter key
            file_path='Image.jpg'
            cv2.imwrite(file_path, img[190:275, 250:385])  # Save the captured image to a file
            break
          elif key == ord('q'):
            # Check if 'q' key is pressed to quit without saving
            print("Quitting without saving.") 
            exit(0)

    # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
        self.display_image(file_path)
        self.predict(file_path)    # Perfrom prediction on captured image
 


    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((135, 85))
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img

    def predict(self, file_path):
        
        model = keras.models.load_model('1.keras')  # loading the model
        image=Image.open(file_path)
        image=image.resize((135,85))   # resizing the captured image according to our model
        image = img_to_array(image)
        image=tf.expand_dims(image,axis=0)  

        prediction = model.predict(image)   # Doing the prediction

        predicted_class=np.argmax(prediction[0])   # Determining the predicted class of ageing

        class_names = ['Fresh', 'Highly Aged', 'Lightly Aged', 'Moderately Aged']
        predicted_class_name=class_names[predicted_class]

        self.result_label.config(text=f"Prediction: {predicted_class_name}")  
        self.send_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = AgeingClassifier(root)
    root.mainloop()