import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
from .canvas_utils import (draw, clear_canvas, create_drawing_surface, reset_brush)
from .inference import predict, DEVICE
from .preprocess import (preprocess_image)



root = tk.Tk()
root.title("MNIST Classifier")

root.geometry("500x1000")
root.configure(bg = "#1e1e1e")


canvas = tk.Canvas(
    root, 
    width = 280,
    height = 280,
    bg = "black",
    highlightthickness = 2,
    highlightbackground = "#444"
)
canvas.pack(pady = 20)

processed_preview = tk.Label(
    root,
    bg = "#1e1e1e"
)
processed_preview.pack(pady = 10)

image, draw_object = create_drawing_surface()


prediction_label = tk.Label(
    root, 
    text = "Draw 0-9",
    font = ("Helvetica", 20, "bold"),
    fg = "white",
    bg = "#1e1e1e",
    justify = "center"
)
prediction_label.pack(pady = 10)


digit_bars = []
digit_labels = []

for digit in range(10):
    label = tk.Label(
        root,
        text = f"{digit}: 0.0%",
        fg = "white",
        bg = "#1e1e1e"
    )
    label.pack()
    bar = ttk.Progressbar(
        root,
        length = 300,
        maximum = 100
    )
    bar.pack(pady = 2)
    digit_labels.append(label)
    digit_bars.append(bar)



def update_prediction():
    tensor, processed_image  = preprocess_image(
        image,
        DEVICE
    )
    predicted_digit, probabilities = predict(tensor)
    confidence = probabilities[predicted_digit] * 100

    preview_image = processed_image.resize((140, 140))
    preview_photo = ImageTk.PhotoImage(preview_image)

    processed_preview.config(
        image = preview_photo
    )
    processed_preview.image = preview_photo


    prediction_label.config(
        text = (
            f"Prediction : {predicted_digit}\n"
            f"Confidence : {confidence:.1f}%"
        )
    )

    for digit in range(10):
        percentage = probabilities[digit] * 100
        digit_bars[digit]["value"] = percentage
        digit_labels[digit].config(text = f"{digit}: {percentage: .1f}")

def reset_prediction():
    prediction_label.config(
        text = "Draw 0-9"
    )
    processed_preview.config(image = "")
    processed_preview.image = None

    for digit in range(10):
        digit_bars[digit]["value"] = 0
        digit_labels[digit].config(
            text = f"{digit}: 0.0%"
        )

    
canvas.bind("<B1-Motion>", lambda event :(draw(event, canvas, draw_object), update_prediction()))
canvas.bind("<ButtonRelease-1>", lambda event : reset_brush())


clear_button = tk.Button(
    root,
    text = "Clear",
    font = ("Helvetica", 14, "bold"),
    bg = "#333",
    fg = "white",
    activebackground = "#555",
    activeforeground = "white",
    relief = "flat",
    padx = 20,
    pady = 5,
    command = lambda :(clear_canvas(canvas, image, draw_object), reset_prediction()) 
)

clear_button.pack(pady = 10)
root.mainloop()

