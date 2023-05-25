import os.path
import tkinter as tk
from screens.inference import Inference
from screens.train import Train
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageDraw


class RoundedButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            relief=tk.RAISED,
            bd=3,
            bg="blue",
            fg="white",
            padx=10,
            pady=10,
            font=("Arial", 20, "bold")
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.configure(bg="white", fg="black")

    def on_leave(self, event):
        self.configure(bg="blue", fg="white")

class Home(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1600x800")
        self.title("inpaintGAN")

        # Create a frame for centering
        center_frame = tk.Frame(self)
        center_frame.pack(pady=50)

        # Create the buttons
        self.button1 = RoundedButton(center_frame, text="Train", command=self.train_click)
        self.button2 = RoundedButton(center_frame, text="Inference", command=self.inference_click)
        self.button3 = RoundedButton(center_frame, text="Datasets", command=self.button3_click)

        self.button1.configure(relief=tk.RAISED, bd=3, bg="blue", fg="white", padx=10, pady=10)
        self.button2.configure(relief=tk.RAISED, bd=3, bg="blue", fg="white", padx=10, pady=10)
        self.button3.configure(relief=tk.RAISED, bd=3, bg="blue", fg="white", padx=10, pady=10)

        self.button1.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.button2.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.button3.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure([0, 1, 2], weight=1, uniform="buttons")

        # Create a label for the title
        self.title_label = tk.Label(center_frame, text="\n\ninpaintGAN \n Generate image inpainting with adversarial edge learning", font=("Calibri", 45, "bold"))
        self.title_label.grid(row=1, column=0, columnspan=3, pady=20)

        # Center the frame within the window
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)


    def inference_click(self):
        self.withdraw()  # Hide the home window
        inference_window = Inference(self)
        inference_window.protocol("WM_DELETE_WINDOW", self.close_inference)

    def close_inference(self):
        self.deiconify()  # Show the home window
        self.focus_force()  # Set focus back to home window

    def train_click(self):
        self.withdraw()  # Hide the home window
        inference_window = Train(self)
        inference_window.protocol("WM_DELETE_WINDOW", self.close_train)

    def close_train(self):
        self.deiconify()  # Show the home window
        self.focus_force()  # Set focus back to home window

    def button3_click(self):
        print("Button 3 clicked")


if __name__ == "__main__":
    gui = Home()
    gui.mainloop()
    # tkinter._test()
