import threading
import time
import tkinter
from enum import Enum, unique
from queue import Queue

from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# an enumeration to decide if continuing visualizing
@unique
class VMsgType(Enum):
	CONTINUE = 'continue'
	STOP = 'stop'


# a message entity to deliver the time matrix
class VisualizerMsg:
	def __init__(self, msg, voxels):
		self.msg = msg
		self.voxels = voxels


class Visualizer:

	# queue: store the delivered time matrix
	def __init__(self, voxelQueue):
		self.queue = voxelQueue
		self.root = tkinter.Tk()
		self.root.wm_title("test")

		self.ax = None
		self.canvas = None
		self.toolbar = None

		self.can_stop = False

	def explode(self, data):
		size = np.array(data.shape) * 2
		data_e = np.zeros(size - 1, dtype=data.dtype)
		data_e[::2, ::2, ::2] = data
		return data_e

	# start visualization
	def start(self):
		# root = tkinter.Tk()
		# root.wm_title("Embedding in Tk")
		plt.axis('off')
		fig = plt.figure()
		self.ax = fig.gca(projection='3d')
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_zticks([])

		v_msg = self.queue.get()
		x, y, z, my_filled, my_fcolor, my_ecolor = self.get_data(v_msg.voxels)
		self.ax.voxels(x, y, z, my_filled, facecolors=my_fcolor, edgecolors=my_ecolor)

		self.canvas = FigureCanvasTkAgg(fig, master=self.root)  # A tk.DrawingArea.
		self.canvas.draw_idle()
		self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

		self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
		self.toolbar.update()
		self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

		self.canvas.mpl_connect("key_press_event", self.on_key_press)

		timer = threading.Timer(1, self.refresh)
		timer.start()

		tkinter.mainloop()

	# refresh the canvas for next visualization
	def refresh(self):
		v_msg = self.queue.get()
		x, y, z, my_filled, my_fcolor, my_ecolor = self.get_data(v_msg.voxels)
		self.ax.clear()
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_zticks([])
		self.ax.voxels(x, y, z, my_filled, facecolors=my_fcolor, edgecolors=my_ecolor)
		self.canvas.draw_idle()
		self.canvas.flush_events()
		if v_msg.msg == VMsgType.STOP:
			self.can_stop = True
		if not self.can_stop:
			timer = threading.Timer(1, self.refresh)
			timer.start()

	def on_key_press(self, event):
		print("you pressed {}".format(event.key))
		key_press_handler(event, self.canvas, self.toolbar)

	# get the data for visualization
	def get_data(self, n_voxels):
		facecolors = np.where(n_voxels, '#0f15bf', '#bf0f15')
		# print(facecolors)
		edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
		filled = np.ones(n_voxels.shape)

		# upscale the above voxel image, leaving gaps
		filled_2 = self.explode(filled)
		fcolors_2 = self.explode(facecolors)
		ecolors_2 = self.explode(edgecolors)

		# Shrink the gaps
		x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
		x[0::2, :, :] += 0.05
		y[:, 0::2, :] += 0.05
		z[:, :, 0::2] += 0.05
		x[1::2, :, :] += 0.95
		y[:, 1::2, :] += 0.95
		z[:, :, 1::2] += 0.95
		return x, y, z, filled_2, fcolors_2, ecolors_2


if __name__ == '__main__':
	queue = Queue(maxsize=0)
	for i in range(4):
		n_voxels = np.zeros([4, 4, 4], dtype=bool)
		n_voxels[0, 0, i] = True
		queue.put(n_voxels)
	queue.put('stop')
	visualizer = Visualizer(queue)
	visualizer.start()
