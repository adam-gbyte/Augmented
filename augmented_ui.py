import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import random

from augmented import (
    random_augment,
    augment_image_file
)

class AugmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Augmentation Tool")
        self.root.geometry("1000x600")

        self.img_path = None
        self.img = None
        self.tk_img = None
        self.preview_img = None

        # -------------------
        # Frame kiri: tombol + parameter
        # -------------------
        control_frame = tk.Frame(root, padx=10, pady=10)
        control_frame.pack(side="left", fill="y")

        btn_load = tk.Button(control_frame, text="Load Image", command=self.load_image)
        btn_load.pack(pady=5, fill="x")

        tk.Label(control_frame, text="Jumlah Augmentasi:").pack(anchor="w")
        self.n_var = tk.IntVar(value=5)
        tk.Entry(control_frame, textvariable=self.n_var).pack(fill="x", pady=2)

        tk.Label(control_frame, text="Rotasi max (°):").pack(anchor="w")
        self.rotate_var = tk.IntVar(value=15)
        tk.Entry(control_frame, textvariable=self.rotate_var).pack(fill="x", pady=2)

        tk.Label(control_frame, text="Zoom range (min,max):").pack(anchor="w")
        self.zoom_min = tk.DoubleVar(value=0.9)
        self.zoom_max = tk.DoubleVar(value=1.2)
        zoom_frame = tk.Frame(control_frame)
        zoom_frame.pack(fill="x", pady=2)
        tk.Entry(zoom_frame, textvariable=self.zoom_min, width=5).pack(side="left")
        tk.Entry(zoom_frame, textvariable=self.zoom_max, width=5).pack(side="left", padx=5)

        tk.Label(control_frame, text="Brightness (min,max):").pack(anchor="w")
        self.bright_min = tk.DoubleVar(value=0.8)
        self.bright_max = tk.DoubleVar(value=1.2)
        bright_frame = tk.Frame(control_frame)
        bright_frame.pack(fill="x", pady=2)
        tk.Entry(bright_frame, textvariable=self.bright_min, width=5).pack(side="left")
        tk.Entry(bright_frame, textvariable=self.bright_max, width=5).pack(side="left", padx=5)

        tk.Label(control_frame, text="Noise std max:").pack(anchor="w")
        self.noise_max = tk.DoubleVar(value=15.0)
        tk.Entry(control_frame, textvariable=self.noise_max).pack(fill="x", pady=2)

        btn_preview = tk.Button(control_frame, text="Preview Augment", command=self.preview)
        btn_preview.pack(pady=10, fill="x")

        btn_save = tk.Button(control_frame, text="Save Augmented Images", command=self.save_augmented)
        btn_save.pack(pady=10, fill="x")

        # -------------------
        # Frame kanan: preview gambar
        # -------------------
        self.canvas = tk.Canvas(root, bg="gray", width=700, height=600)
        self.canvas.pack(side="right", fill="both", expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return
        self.img_path = path
        self.img = Image.open(path).convert("RGB")
        self.show_image(self.img)

    def show_image(self, img):
        img_resized = img.copy()
        img_resized.thumbnail((700, 600))
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(350, 300, image=self.tk_img)

    def preview(self):
        if self.img is None:
            messagebox.showwarning("No Image", "Silakan load gambar dulu.")
            return
        aug = random_augment(
            self.img,
            rotate_range=(-self.rotate_var.get(), self.rotate_var.get()),
            zoom_range=(self.zoom_min.get(), self.zoom_max.get()),
            brightness_range=(self.bright_min.get(), self.bright_max.get()),
            contrast_range=(0.8, 1.2),
            noise_std_range=(0.0, self.noise_max.get())
        )
        self.preview_img = aug
        self.show_image(aug)

    def save_augmented(self):
        if self.img is None:
            messagebox.showwarning("No Image", "Silakan load gambar dulu.")
            return
        outdir = filedialog.askdirectory(title="Pilih Folder Output")
        if not outdir:
            return
        n = self.n_var.get()
        augment_image_file(self.img_path, outdir, n=n)
        messagebox.showinfo("Done", f"{n} gambar augmented disimpan ke {outdir}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AugmentApp(root)
    root.mainloop()
