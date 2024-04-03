

import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import re


class MaskCreator:
    def __init__(self, root):
        self.root = root
        self.root.title('01 Mask Creator')

        # 设定画布大小
        self.canvas_width = 500  # 更新为400x400，根据需求调整
        self.canvas_height = 500
        self.canvas = tk.Canvas(root, cursor="cross", width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image = None
        self.photo = None
        self.draw = None
        self.original_size = None  # 保存原始图像尺寸
        self.file_path = None  # 保存文件路径
        self.brush_size = 12

        # Bind canvas events
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        frame_buttons = tk.Frame(root)
        frame_buttons.pack(fill=tk.X)

        btn_open = tk.Button(frame_buttons, text="Open Image", command=self.open_image)
        btn_open.pack(side=tk.LEFT, padx=5)

        btn_save = tk.Button(frame_buttons, text="Save Mask", command=self.save_mask)
        btn_save.pack(side=tk.LEFT, padx=5)

        btn_brush_size = tk.Scale(frame_buttons, from_=1, to=50, orient=tk.HORIZONTAL, label="Brush Size")
        btn_brush_size.set(self.brush_size)
        btn_brush_size.pack(side=tk.LEFT, padx=5)
        btn_brush_size.bind("<Motion>", self.change_brush_size)

        # 路径显示标签
        self.path_label = Label(root, text="save_path=", wraplength=self.canvas_width)
        self.path_label.pack(side=tk.BOTTOM, fill=tk.X)

    def open_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.image = Image.open(self.file_path).convert("RGBA")
            self.original_size = self.image.size  # 保存原始图像尺寸
            # 将图像缩放到设定的画布大小
            self.image = self.image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
            self.mask = Image.new('L', (self.canvas_width, self.canvas_height), 0)
            self.draw = ImageDraw.Draw(self.mask)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            # 更新路径显示
            self.path_label.config(text=f"save_path={self.file_path}")

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill="lightblue", outline="lightblue")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def save_mask(self):
        if self.image and self.file_path:
            # 生成保存路径
            directory, filename = os.path.split(self.file_path)
            basename, ext = os.path.splitext(filename)
            mask_filename = f"{basename}_mask.png"  # 基本文件名
            save_path = os.path.join(directory, mask_filename)
            # 检查文件是否存在，如果存在则增加数字后缀
            counter = 1
            while os.path.exists(save_path):
                mask_filename = f"{basename}_mask_{counter}.png"
                save_path = os.path.join(directory, mask_filename)
                counter += 1
            # 将mask重新缩放回原始图像尺寸
            mask_resized = self.mask.resize(self.original_size, Image.Resampling.LANCZOS)
            mask_np = np.array(mask_resized)
            mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_np)
            mask_image.save(save_path)
            # 显示保存路径
            self.path_label.config(text=f"Mask saved to: {save_path}")

    def change_brush_size(self, event):
        self.brush_size = event.widget.get()


if __name__ == "__main__":
    root = tk.Tk()
    app = MaskCreator(root)
    root.mainloop()