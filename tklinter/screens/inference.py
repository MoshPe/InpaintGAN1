import os.path
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageDraw


class Inference(tk.Toplevel):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.geometry("1600x1300")
        self.create_widgets()
        self.eraser_active = False  # Flag to keep track of eraser button state
        self.image_loaded = False  # Flag to keep track of whether an image is loaded
        self.mask = None
        self.mask_photo = None
        self.image = None
        self.photo = None
        # self.config = Config()
        # self.config.MODEL = 4
        # self.edgeConnect = EdgeConnect(self.config)

        # Variables to store mouse coordinates
        self.start_x = None
        self.start_y = None

    def create_widgets(self):

        eraser_icon = Image.open(
            "screens/assets/eraser_icon.png")  # Replace "eraser_icon.png" with the actual file path of the icon image
        eraser_icon = eraser_icon.resize((32, 32))  # Resize the icon to desired dimensions
        self.eraser_image = ImageTk.PhotoImage(eraser_icon)

        home_icon = Image.open(
            "screens/assets/home_btn.png")  # Replace "eraser_icon.png" with the actual file path of the icon image
        home_icon = home_icon.resize((60, 60))  # Resize the icon to desired dimensions
        self.home_image = ImageTk.PhotoImage(home_icon)

        self.home_button = tk.Button(self, image=self.home_image, width=60, height=60, borderwidth=0, command=self.close,
                                       bd=0, highlightthickness=0, activebackground="gray")
        self.home_button.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.N)

        self.eraser_button = tk.Button(self, image=self.eraser_image, width=60, height=60, command=self.toggle_eraser,
                                       bd=1, highlightthickness=0, activebackground="gray")
        self.eraser_button.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.N)

        self.image_label = tk.Label(self)
        self.image_label.pack(side=tk.LEFT, pady=10, padx=(0, 20), anchor=tk.NE)

        self.mask_label = tk.Label(self)
        self.mask_label.pack(side=tk.RIGHT, pady=10, padx=(20, 0), anchor=tk.NW)

        self.error_label = tk.Label(self, fg="red")
        self.error_label.pack(side=tk.BOTTOM, pady=10)

        self.fill_button = tk.Button(self, text="Fill Image", width=20, height=2, command=self.fill_image)
        self.fill_button.pack(side=tk.BOTTOM, pady=10, anchor=tk.S)

        self.upload_button = tk.Button(self, text="Upload Image", width=20, height=2, command=self.upload_image)
        self.upload_button.pack(side=tk.BOTTOM, pady=10, anchor=tk.S)

        self.name_label = tk.Label(self, text="")
        self.name_label.pack(side=tk.BOTTOM, pady=10, anchor=tk.S)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            self.image = Image.open(file_path)
            width, height = self.image.size
            if width > 600 or height > 800:
                ratio = min(600 / width, 800 / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.ANTIALIAS)

            self.mask = Image.new("L", self.image.size)

            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

            self.mask_photo = ImageTk.PhotoImage(self.mask)
            self.mask_label.config(image=self.mask_photo)
            self.mask_label.image = self.mask_photo

            self.name_label.config(text="File Name: " + os.path.basename(file_path))
            self.error_label.config(text="")
            self.image_loaded = True

    def fill_image(self):
        if self.image_loaded:
            result = self.edgeConnect.fill_image(self.image, self.mask)
            self.photo = ImageTk.PhotoImage(result)
            self.image_label.config(image=self.photo)
        else:
            self.error_label.config(text="Error: No image uploaded")

    def toggle_eraser(self):
        self.eraser_active = not self.eraser_active  # Toggle eraser button state
        if self.eraser_active:
            # Change mouse cursor to eraser icon
            self.config(cursor="spraycan")
            self.eraser_button.configure(background="gray")
            self.image_label.bind("<Button-1>", self.start_drawing)
            self.image_label.bind("<B1-Motion>", self.erase_pixel)
            self.image_label.bind("<ButtonRelease-1>", self.stop_drawing)
        else:
            # Change mouse cursor back to normal and eraser button background to white
            self.config(cursor="")
            self.eraser_button.configure(background="white")
            self.image_label.unbind("<Button-1>")
            self.image_label.unbind("<B1-Motion>")
            self.image_label.unbind("<ButtonRelease-1>")

    def start_drawing(self, event):
        if self.image_loaded:
            self.start_x = event.x
            self.start_y = event.y

    def stop_drawing(self, event):
        if self.image_loaded:
            self.start_x = None
            self.start_y = None

    def erase_pixel(self, event):
        if self.image_loaded and self.start_x is not None and self.start_y is not None:
            draw = ImageDraw.Draw(self.image)
            draw.line((self.start_x, self.start_y, event.x, event.y), fill="#ffffff", width=10)

            mask_draw = ImageDraw.Draw(self.mask)
            mask_draw.line((self.start_x, self.start_y, event.x, event.y), fill="#ffffff", width=10)

            self.mask_photo = ImageTk.PhotoImage(self.mask)
            self.mask_label.config(image=self.mask_photo)

            # Update self.photo with the modified image
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)

            # Update starting position to the current position
            self.start_x = event.x
            self.start_y = event.y

    def close(self):
        self.destroy()  # Close the inference window
        self.master.deiconify()  # Show the home window
        self.master.focus_force()  # Set focus back to home window
