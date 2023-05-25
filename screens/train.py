import os.path
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageDraw


class Train(tk.Toplevel):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.geometry("1600x800")
        self.title("training")
        self.create_widgets()

        home_icon = Image.open(
            "screens/assets/home_btn.png")  # Replace "eraser_icon.png" with the actual file path of the icon image
        home_icon = home_icon.resize((60, 60))  # Resize the icon to desired dimensions
        self.home_image = ImageTk.PhotoImage(home_icon)

        self.label = tk.Label(self, text="Training Process", font=("Calibri", 40, "bold"))
        self.label.place(relx=0.5, rely=0.1, anchor="center")

        self.home_button = tk.Button(self, image=self.home_image, width=60, height=60, borderwidth=0, padx=20, pady=20,
                                     command=self.back_to_home,
                                     bd=0, highlightthickness=0, activebackground="gray")
        self.home_button.grid(row=0, column=0, sticky="nw")

    def create_widgets(self):
        title1 = tk.Label(self, text="Column 1", font=("Arial", 12, "bold"))
        title1.grid(row=1, column=0, columnspan=2, pady=(0, 10),  sticky="w", padx=10)
        title2 = tk.Label(self, text="Column 2", font=("Arial", 12, "bold"))
        title2.grid(row=1, column=4, columnspan=2, pady=(0, 10),  sticky="w")
        # Add empty columns
        # self.grid_columnconfigure([0, 6], weight=1)

        # Dropdown button 1
        self.var1 = tk.StringVar(self)
        self.var1.set("Option 1")
        self.dropdown1 = tk.OptionMenu(self, self.var1, "Option 1", "Option 2", "Option 3")
        self.dropdown1.grid(row=2, column=1, sticky="w")
        self.text1 = tk.Label(self, text="Text 1", font=("Arial", 20))
        self.text1.grid(row=2, column=0, sticky="w", padx=10)

        # Dropdown button 2
        self.var2 = tk.StringVar(self)
        self.var2.set("Option 2")
        self.dropdown2 = tk.OptionMenu(self, self.var2, "Option 1", "Option 2", "Option 3")
        self.dropdown2.grid(row=3, column=1, sticky="w")
        self.text2 = tk.Label(self, text="Text 2", font=("Arial", 20))
        self.text2.grid(row=3, column=0, sticky="w", padx=10)

        # Dropdown button 3
        self.var3 = tk.StringVar(self)
        self.var3.set("Option 3")
        self.dropdown3 = tk.OptionMenu(self, self.var3, "Option 1", "Option 2", "Option 3")
        self.dropdown3.grid(row=4, column=1, sticky="w")
        self.text3 = tk.Label(self, text="Text 3", font=("Arial", 20))
        self.text3.grid(row=4, column=0, sticky="w", padx=10)

        # Dropdown button 4
        self.var4 = tk.StringVar(self)
        self.var4.set("Option 4")
        self.dropdown4 = tk.OptionMenu(self, self.var4, "Option 4", "Option 5", "Option 6")
        self.dropdown4.grid(row=2, column=5, sticky="w")
        self.text4 = tk.Label(self, text="Text 4", font=("Arial", 20))
        self.text4.grid(row=2, column=4, sticky="w")

        # Dropdown button 5
        self.var5 = tk.StringVar(self)
        self.var5.set("Option 5")
        self.dropdown5 = tk.OptionMenu(self, self.var5, "Option 4", "Option 5", "Option 6")
        self.dropdown5.grid(row=3, column=5, sticky="w")
        self.text5 = tk.Label(self, text="Text 5", font=("Arial", 20))
        self.text5.grid(row=3, column=4, sticky="w")

        # Dropdown button 6
        self.var6 = tk.StringVar(self)
        self.var6.set("Option 6")
        self.dropdown6 = tk.OptionMenu(self, self.var6, "Option 4", "Option 5", "Option 6")
        self.dropdown6.grid(row=4, column=5, sticky="w")
        self.text6 = tk.Label(self, text="Text 6", font=("Arial", 20))
        self.text6.grid(row=4, column=4, sticky="w")



        self.space1 = tk.Label(self, text="                ", font=("Arial", 20))
        self.space1.grid(row=2, column=3, sticky="w")


        self.space2 = tk.Label(self, text="                ", font=("Arial", 20))
        self.space2.grid(row=3, column=3, sticky="w")


        self.space3 = tk.Label(self, text="                ", font=("Arial", 20))
        self.space3.grid(row=4, column=3, sticky="w")

        self.right()
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure([0, 2, 4, 6, 8], minsize=20)

    def right(self):
        # Title labels
        title1 = tk.Label(self, text="Column 1", font=("Arial", 12, "bold"))
        title1.grid(row=1, column=7, columnspan=2, pady=(0, 10), sticky="w", padx=10)
        title2 = tk.Label(self, text="Column 2", font=("Arial", 12, "bold"))
        title2.grid(row=1, column=10, columnspan=2, pady=(0, 10), sticky="w")

        # Dropdown button 1
        var1 = tk.StringVar(self)
        var1.set("Option 1")
        dropdown1 = tk.OptionMenu(self, var1, "Option 1", "Option 2", "Option 3")
        dropdown1.grid(row=2, column=8, sticky="w")
        text1 = tk.Label(self, text="Text 1", font=("Arial", 20))
        text1.grid(row=2, column=7, sticky="w", padx=10)

        # Dropdown button 2
        var2 = tk.StringVar(self)
        var2.set("Option 2")
        dropdown2 = tk.OptionMenu(self, var2, "Option 1", "Option 2", "Option 3")
        dropdown2.grid(row=3, column=8, sticky="w")
        text2 = tk.Label(self, text="Text 2", font=("Arial", 20))
        text2.grid(row=3, column=7, sticky="w", padx=10)

        # Dropdown button 3
        var3 = tk.StringVar(self)
        var3.set("Option 3")
        dropdown3 = tk.OptionMenu(self, var3, "Option 1", "Option 2", "Option 3")
        dropdown3.grid(row=4, column=8, sticky="w")
        text3 = tk.Label(self, text="Text 3", font=("Arial", 20))
        text3.grid(row=4, column=7, sticky="w", padx=10)

        # Dropdown button 4
        var4 = tk.StringVar(self)
        var4.set("Option 4")
        dropdown4 = tk.OptionMenu(self, var4, "Option 4", "Option 5", "Option 6")
        dropdown4.grid(row=2, column=11, sticky="w")
        text4 = tk.Label(self, text="Text 4", font=("Arial", 20))
        text4.grid(row=2, column=10, sticky="w")

        # Dropdown button 5
        var5 = tk.StringVar(self)
        var5.set("Option 5")
        dropdown5 = tk.OptionMenu(self, var5, "Option 4", "Option 5", "Option 6")
        dropdown5.grid(row=3, column=11, sticky="w")
        text5 = tk.Label(self, text="Text 5", font=("Arial", 20))
        text5.grid(row=3, column=10, sticky="w")

        # Dropdown button 6
        var6 = tk.StringVar(self)
        var6.set("Option 6")
        dropdown6 = tk.OptionMenu(self, var6, "Option 4", "Option 5", "Option 6")
        dropdown6.grid(row=4, column=11, sticky="w")
        text6 = tk.Label(self, text="Text 6", font=("Arial", 20))
        text6.grid(row=4, column=10, sticky="w")

        self.space1 = tk.Label(self, text="                ", font=("Arial", 20))
        self.space1.grid(row=2, column=9, sticky="w")


        self.space2 = tk.Label(self, text="                ", font=("Arial", 20))
        self.space2.grid(row=3, column=9, sticky="w")


        self.space3 = tk.Label(self, text="                ", font=("Arial", 20))
        self.space3.grid(row=4, column=9, sticky="w")

    def back_to_home(self):
        self.destroy()  # Close the inference window
        self.master.deiconify()  # Show the home window
        self.master.focus_force()  # Set focus back to home window