from tkinter import *
from tkinter.ttk import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter
from tkinter.filedialog import askopenfile
import torch
import torchvision
import numpy as np
from PIL import Image, ImageTk
from inference import inference
from torchvision.io import read_image

root = Tk()
root.title('Сегментація COVID-19')
root.geometry('700x500')

def open_file():
    ''' open a file and call backend interface '''
    file = askopenfile(mode='r')
    filename = file.name.split('/')[-1]
    ct = read_image('./examples/' + filename)
    mask = inference('./examples/' + filename)

    ct = (torch.tensor(ct)).to(dtype=torch.uint8)
    resize = torchvision.transforms.Resize(224, antialias=True)
    ct = resize(ct)
    mask = mask.permute(2,0,1)[0]

    masked_image = torchvision.utils.draw_segmentation_masks(ct, mask, colors=['red'])

    plt.imshow(masked_image.permute(1, 2, 0))

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(ct.permute(1,2,0))
    axarr[0].set_title("Слайс зображення КТ", fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axarr[1].imshow(mask)
    axarr[1].set_title("Маска сегментації", fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axarr[2].imshow(masked_image.permute(1,2,0))
    axarr[2].set_title("Візуалізація сегментації", fontdict={'fontsize': 8, 'fontweight': 'medium'})

    axarr[0].set_xticklabels([])
    axarr[0].set_yticklabels([])
    axarr[1].set_xticklabels([])
    axarr[1].set_yticklabels([])
    axarr[2].set_xticklabels([])
    axarr[2].set_yticklabels([])

    axarr[0].tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False)
    axarr[1].tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False)
    axarr[2].tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False)

    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()


btn = Button(root, text='Вибрати файл', command=lambda: open_file())
btn.pack(side=TOP, pady=10)

mainloop()