from tkinter import *
from PIL import Image,ImageTk
import numpy as np
import cv2
from number_prediction_service import NumberPredictionService

app = Tk()
app.geometry("400x400")

canvas = np.ones((280, 280), dtype=np.uint8) * 255
print("\nDraw a digit on the image and it will predict the number.")
print("Use an image editing tool to draw, then upload the image.")
# Save and display the blank canvas (so you can draw on it)
filename = "draw_digit.png"
cv2.imwrite(filename, canvas)

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_smth(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), fill='black', width=5)
    lasx, lasy = event.x, event.y
    

canvas = Canvas(app, bg='black')
canvas.pack(anchor='nw', fill='both', expand=1)

canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)


image_origin = Image.open("draw_digit.png")
image_origin = image_origin.resize((400,400), Image.Resampling.LANCZOS)
image_tk = ImageTk.PhotoImage(image_origin)
canvas.create_image(0,0, image=image_tk, anchor='nw')
image_origin.save("draw_digit.png")
predict = Button(app, text="Predict", command= lambda: NumberPredictionService.predict("draw_digit.png"))
predict.pack()

cancel = Button(app, text="Cancel",command=quit)
cancel.pack()


app.mainloop()