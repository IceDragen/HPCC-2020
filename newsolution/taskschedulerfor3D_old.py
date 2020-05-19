import datetime
import math
import operator as op
import os
import random
import sys
import time
from queue import Queue, PriorityQueue
import heapq

import numpy as np

from newsolution.Entity import FitMethodType, Task, RunningRecord


class OldTaskSchedulerFor3D:
	# size: 矩阵规模，是个list
	# file_path: 存储task数据的文件路径
	# taskQueue: 任务存放队列
	# hpc: hpc矩阵
	# task_num: 需要模拟的任务的数量
	# arrival_rate: 柏松过程中的lambda，即每秒到达的平均任务数
	# lock: 由于安排任务和任务时间减一都需要修改hpc矩阵，所以要用锁保证线程安全
	# remaining_task_num: 剩余还没有被安排的任务数量，作为消费者和定时任务停止的判定条件之一
	# scheduler： 定时任务调度器
	# time_job_id: 定时任务的名字
	def __init__(self, size, task_queue, data_path=None, time_path=None, arrival_rate=2, method_name=FitMethodType.FIRST_FIT,
	             enable_back_filling=False, enable_visualize=False, st=1, enable_time_sort=False):
		self.enable_back_filling = enable_back_filling
		self.taskQueue = task_queue
		self.voxelQueue = Queue(maxsize=0)
		self.first_display = True
		self.enable_visualize = enable_visualize
		self.run_out_task_num = 0
		self.has_scheduled_task_num = 0
		self.wasted_nodes = 0
		self.util_data_file_path = data_path
		self.time_data_file_path = time_path
		self.enable_sort = enable_time_sort

		self.hpc = np.zeros(size, dtype=int)
		self.job_name_matrix = np.zeros(size, dtype=int)
		self.total_nodes = size[0] * size[1] * size[2]
		self.empty_nodes = self.total_nodes

		self.method_name = method_name
		# 记录目前已经安排了几个j1520-j3530之间的job
		self.target_fit_job_num = 0
		self.can_program_stop = False

		self.running_task_list = []
		self.running_task_ids = set()

		self.time_counter = 0
		self.start_record_util = False
		self.start_time = None
		self.end_time = None
		self.has_record_data = False
		self.has_begin_time_count = False

		self.u5 = []
		self.u10 = []
		self.u15 = []

		# for online
		self.prototype_queue = self.taskQueue
		self.poisson_task_queue = self.taskQueue[:1520]
		self.has_left_task_to_fetch = True
		self.time_strategy = st
		# 指向下一次可以拿任务的位置
		self.prototype_queue_cursor = 1520
		# 指向下一次可以拿任务的位置
		self.poisson_queue_cursor = 0
		self.arrival_rate = arrival_rate

		self.sorted_queue = []

		# 本次back_filling结束倒计时
		self.counter = 0
		self.last_location = (0, 0, 0)
		# self.test_set = set()

		self.visualizer = None

	def test_generate_task(self):
		file_path = './data/task.txt'
		f = open(file_path)
		lines = f.readlines()
		f.close()
		for line in lines:
			values = line.split(',')
			task = Task(values[0], int(values[1]), int(values[2]))
			self.taskQueue.append(task)

	# 初始化job name矩阵
	def job_name_matrix_init(self):
		[height, rows, columns] = self.job_name_matrix.shape
		for i in range(rows):
			for j in range(columns):
				for h in range(height):
					self.job_name_matrix[i, j] = 0

	# 生产任务
	def generate_task(self):
		i = 1
		for t in range(20):
			j_name = 'j' + str(i)
			volume = random.randint(1, int(self.total_nodes/2))
			cost_time = random.randint(1, 10)
			task = Task(j_name, volume, cost_time)
			self.taskQueue.append(task)
			i = i + 1
		self.poisson_task_queue = self.taskQueue[:5]
		# 指向下一次可以拿任务的位置
		self.prototype_queue_cursor = 5
		# 指向下一次可以拿任务的位置
		self.poisson_queue_cursor = 0

	def has_run(self, task):
		return task.name in self.running_task_ids

	def print_matrix(self):
		[height, rows, columns] = self.hpc.shape
		for h in range(height - 1, -1, -1):
			for i in range(rows - 1, -1, -1):
				for j in range(columns):
					print('%2d' % self.hpc[h, i, j], end=' ')
				print(end='\t')
				for j in range(columns):
					print('%2d' % self.job_name_matrix[h, i, j], end=' ')
				print()
			print()

	def schedule(self, method_name, task, time_matrix, last_location=None, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		is_find = False
		locations = []

		if method_name == FitMethodType.FIRST_FIT:
			is_find, locations = self.first_fit(task, time_matrix, is_back_filling_task, waiting_time,
			                                    preserve_locations)
		elif method_name == FitMethodType.BEST_FIT:
			is_find, locations = self.best_fit(task, time_matrix, is_back_filling_task, waiting_time, preserve_locations)
		elif method_name == FitMethodType.WORST_FIT:
			is_find, locations = self.worst_fit(task, time_matrix, is_back_filling_task, waiting_time,
			                                    preserve_locations)
		elif method_name == FitMethodType.NEXT_FIT:
			is_find, locations = self.next_fit(task, time_matrix, last_location, is_back_filling_task, waiting_time,
			                                   preserve_locations)

		return is_find, locations

	# 每个已安排任务的可执行时间减一
	def time_process(self):
		changed = False
		need_visualize_change = False

		[height, rows, columns] = self.hpc.shape
		for h in range(height):
			for i in range(rows):
				for j in range(columns):
					if self.hpc[h, i, j] > 0:
						self.hpc[h, i, j] = self.hpc[h, i, j] - 1
						changed = True
					if self.hpc[h, i, j] == 0:
						self.job_name_matrix[h, i, j] = 0
						if self.enable_visualize:
							need_visualize_change = True

		if self.enable_back_filling:
			if self.counter > 0:
				self.counter = self.counter - 1

		self.check_and_update_record()

	# if self.enable_visualize and need_visualize_change:
	# 	self.post_process_voxels()

	# if self.remaining_task_num == 0 and not changed:
	# 	print("已完成所有任务")
	# 	self.stop_visualize()

	# def stop_visualize(self):
	# 	if self.enable_visualize:
	# 		vMsg = OldVisualizerMsg(VMsgType.STOP, self.hpc, [])
	# 		self.voxelQueue.put(vMsg)

	def check_and_update_record(self):
		removing_id_list = []
		for i, r in enumerate(self.running_task_list):
			if r.rest_time - 1 <= 0:
				removing_id_list.append(i)
				self.empty_nodes = self.empty_nodes + r.volume
				self.wasted_nodes = self.wasted_nodes - len(r.wasted_nodes_locations)
			else:
				r.rest_time = r.rest_time - 1

		self.run_out_task_num = self.run_out_task_num + len(removing_id_list)

		if len(removing_id_list) > 0:
			new_running_tasks = [t for i, t in enumerate(self.running_task_list) if i not in removing_id_list]
			self.running_task_list = new_running_tasks

	def check_if_adjust_record_util(self):
		# 上限3530
		if 'j3530' in self.running_task_ids and self.target_fit_job_num == 2010:
			self.start_record_util = False
			if not self.has_record_data:
				self.end_time = time.time()
				self.write_utilization_to_file()
				self.write_cost_time_to_file()
				self.has_record_data = True
				self.can_program_stop = True

		# 下限1520
		if 'j1520' in self.running_task_ids:
			self.start_record_util = True
			if not self.has_begin_time_count:
				self.start_time = time.time()
				self.has_begin_time_count = True

	def write_utilization_to_file(self):
		delimiter = ','
		data5 = delimiter.join(self.u5)
		data10 = delimiter.join(self.u10)
		data15 = delimiter.join(self.u15)
		all_data = [data5]
		# for str in all_data:
		# 	print(all_data)

		with open(self.util_data_file_path, 'a+') as f:
			for data in all_data:
				f.write(data)
				f.write('\n')

	def write_cost_time_to_file(self):
		cost_time = self.end_time - self.start_time
		with open(self.time_data_file_path, 'a+') as f:
			f.write(str(cost_time))
			f.write('\n')

	def do_after_find(self, task, locations):
		# 如果已经找到，那么将对应的Nodes的数值赋为任务所需的执行时间

		self.update_hpc(locations, task)
		self.check_if_adjust_record_util()
		# print('当前正在安排{}, 需要{}个nodes, 该task需要花费时间为{}个单位时间:'.format(task.name, task.volume, task.time))
		# self.print_matrix()

	# 左右移动的面
	def get_xoz_sum_matrix(self, matrix):
		result_matrix = np.zeros(matrix.shape, dtype=int)
		(height, rows, columns) = matrix.shape
		for j in range(columns):
			result_matrix[0, 0, j] = matrix[0, 0, j]
			for h in range(1, height):
				result_matrix[h, 0, j] = result_matrix[h - 1, 0, j] + matrix[h, 0, j]
			for i in range(1, rows):
				result_matrix[0, i, j] = result_matrix[0, i - 1, j] + matrix[0, i, j]
			for h in range(1, height):
				for i in range(1, rows):
					result_matrix[h, i, j] = result_matrix[h - 1, i, j] + result_matrix[h, i - 1, j] - result_matrix[
						h - 1, i - 1, j] + matrix[h, i, j]
		return result_matrix

	# j不变，h和i变
	def get_xoz_matrix_sum(self, sum_matrix, start_location, end_location):
		(start_h, start_i, start_j) = start_location
		(end_h, end_i, end_j) = end_location
		j = start_j
		a = b = c = d = 0
		a1, a2 = start_h - 1, start_i - 1
		b1, b2 = start_h - 1, end_i
		c1, c2 = end_h, start_i - 1
		d1, d2 = end_h, end_i
		if a1 < 0 or a2 < 0:
			a = 0
		else:
			a = sum_matrix[a1, a2, j]
		if b1 < 0 or b2 < 0:
			b = 0
		else:
			b = sum_matrix[b1, b2, j]
		if c1 < 0 or c2 < 0:
			c = 0
		else:
			c = sum_matrix[c1, c2, j]
		if d1 < 0 or d2 < 0:
			d = 0
		else:
			d = sum_matrix[d1, d2, j]
		return a + d - b - c

	# 前后移动的面
	def get_yoz_sum_matrix(self, matrix):
		result_matrix = np.zeros(matrix.shape, dtype=int)
		(height, rows, columns) = matrix.shape
		for i in range(rows):
			result_matrix[0, i, 0] = matrix[0, i, 0]
			for j in range(1, columns):
				result_matrix[0, i, j] = result_matrix[0, i, j - 1] + matrix[0, i, j]
			for h in range(1, height):
				result_matrix[h, i, 0] = result_matrix[h - 1, i, 0] + matrix[h, i, 0]
			for j in range(1, columns):
				for h in range(1, height):
					result_matrix[h, i, j] = result_matrix[h - 1, i, j] + result_matrix[h, i, j - 1] - result_matrix[
						h - 1, i, j - 1] + matrix[h, i, j]
		return result_matrix

	# i不变，h和j变
	def get_yoz_matrix_sum(self, sum_matrix, start_location, end_location):
		(start_h, start_i, start_j) = start_location
		(end_h, end_i, end_j) = end_location
		i = start_i
		a = b = c = d = 0
		a1, a2 = start_h - 1, start_j - 1
		b1, b2 = start_h - 1, end_j
		c1, c2 = end_h, start_j - 1
		d1, d2 = end_h, end_j
		if a1 < 0 or a2 < 0:
			a = 0
		else:
			a = sum_matrix[a1, i, a2]
		if b1 < 0 or b2 < 0:
			b = 0
		else:
			b = sum_matrix[b1, i, b2]
		if c1 < 0 or c2 < 0:
			c = 0
		else:
			c = sum_matrix[c1, i, c2]
		if d1 < 0 or d2 < 0:
			d = 0
		else:
			d = sum_matrix[d1, i, d2]
		return a + d - b - c

	def get_auxiliary_data(self, matrix):
		height = np.zeros(matrix.shape, dtype=int)
		depth_in = np.zeros(matrix.shape, dtype=int)
		depth_out = np.full(matrix.shape, matrix.shape[1], dtype=int)
		left = np.zeros(matrix.shape, dtype=int)
		right = np.full(matrix.shape, matrix.shape[2], dtype=int)
		(talls, rows, columns) = matrix.shape
		xoz_sum_matrix = self.get_xoz_sum_matrix(matrix)
		yoz_sum_matrix = self.get_yoz_sum_matrix(matrix)
		for h in range(talls):
			for i in range(rows):
				for j in range(columns):
					# 向下找线
					if matrix[h, i, j] == 0:
						if h - 1 < 0:
							height[h, i, j] = 1
						else:
							height[h, i, j] = height[h - 1, i, j] + 1
					else:
						height[h, i, j] = 0

					# 根据线前后找面
					if matrix[h, i, j] == 0:
						last_in = last_out = i
						start_h, start_j = h - height[h, i, j] + 1, j
						end_h, end_j = h, j
						for i1 in range(i - 1, -1, -1):
							start_location = (start_h, i1, start_j)
							end_location = (end_h, i1, end_j)
							tmp = self.get_yoz_matrix_sum(yoz_sum_matrix, start_location, end_location)
							if tmp == 0:
								last_out = last_out - 1
							else:
								break
						depth_out[h, i, j] = last_out

						for i1 in range(i + 1, rows):
							start_location = (start_h, i1, start_j)
							end_location = (end_h, i1, end_j)
							tmp = self.get_yoz_matrix_sum(yoz_sum_matrix, start_location, end_location)
							if tmp == 0:
								last_in = last_in + 1
							else:
								break
						depth_in[h, i, j] = last_in
					else:
						depth_out[h, i, j] = -1
						depth_in[h, i, j] = -1

					# 根据面左右找体
					if matrix[h, i, j] == 0:
						start_h, start_i = h - height[h, i, j] + 1, depth_out[h, i, j]
						end_h, end_i = h, depth_in[h, i, j]
						last_left = last_right = j
						for j1 in range(j - 1, -1, -1):
							start_location = (start_h, start_i, j1)
							end_location = (end_h, end_i, j1)
							tmp = self.get_xoz_matrix_sum(xoz_sum_matrix, start_location, end_location)
							if tmp == 0:
								last_left = last_left - 1
							else:
								break
						left[h, i, j] = last_left
						for j1 in range(j + 1, columns):
							start_location = (start_h, start_i, j1)
							end_location = (end_h, end_i, j1)
							tmp = self.get_xoz_matrix_sum(xoz_sum_matrix, start_location, end_location)
							if tmp == 0:
								last_right = last_right + 1
							else:
								break
						right[h, i, j] = last_right
					else:
						left[h, i, j] = -1
						right[h, i, j] = -1
		return height, depth_out, depth_in, left, right

	def get_volumes_between_two_points(self, start_location, end_location):
		(h1, i1, j1) = start_location
		(h2, i2, j2) = end_location
		return (h2 - h1 + 1) * (i2 - i1 + 1) * (j2 - j1 + 1)

	def check_volumes(self, start_location, end_location, task):
		volumes = self.get_volumes_between_two_points(start_location, end_location)
		return volumes >= task.volume

	def get_all_locations_between_two_points(self, start_location, end_location):
		(h1, i1, j1) = start_location
		(h2, i2, j2) = end_location
		locations = []
		for h in range(h1, h2 + 1):
			for i in range(i1, i2 + 1):
				for j in range(j1, j2 + 1):
					locations.append((h, i, j))
		return locations

	# 找到的矩阵可能很大，任务其中一个子矩阵就够了，所以要找到放任务的精确位置
	def get_precise_start_and_end_locations_for_task(self, task, start_location, end_location):
		(start_h, start_i, start_j) = start_location
		(end_h, end_i, end_j) = end_location
		long = end_i - start_i + 1
		width = end_j - start_j + 1
		s = long * width
		tmp = 0
		if width >= task.volume:
			tmp = task.volume
			over_location = (start_h, start_i, start_j + tmp - 1)
			# print('one line: {},{}'.format(tmp, len(locations)))
			return start_location, over_location
		elif s >= task.volume:
			tmp = math.ceil(task.volume / width)
			# print(task)
			# print('one mian: {},{}'.format(tmp, len(locations)))
			return start_location, (start_h, start_i + tmp - 1, end_j)
		else:
			tmp = math.ceil(task.volume / s)
			# print('one changfanti: {},{}'.format(tmp, len(locations)))
			return start_location, (start_h + tmp - 1, end_i, end_j)

	def judge_in_middle(self, start_location, end_location, judge_location):
		(start_h, start_i, start_j) = start_location
		(end_h, end_i, end_j) = end_location
		(j_h, j_i, j_j) = judge_location
		a = start_h <= j_h <= end_h
		b = start_i <= j_i <= end_i
		c = start_j <= j_j <= end_j
		return a and b and c

	def check_if_back_filling_task_can_put(self, start_location, end_location, preserve_locations, task, waiting_time):
		# 判断预留的位置是否和找出来的位置有交集
		r1 = self.judge_in_middle(preserve_locations[0], preserve_locations[-1], start_location)
		r2 = self.judge_in_middle(preserve_locations[0], preserve_locations[-1], end_location)
		if r1 or r2:
			if task.time <= waiting_time:
				return True
			else:
				return False
		return True

	# 用first fit策略为一个任务分配Nodes
	def first_fit(self, task, time_matrix, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		height, depth_out, depth_in, left, right = self.get_auxiliary_data(time_matrix)
		is_find, locations = False, []
		(talls, rows, columns) = self.hpc.shape
		flag = False
		start_location = end_location = (0, 0, 0)
		for h in range(talls):
			for i in range(rows):
				for j in range(columns):
					h0 = height[h, i, j]
					l0 = right[h, i, j] - left[h, i, j] + 1
					w0 = depth_in[h, i, j] - depth_out[h, i, j] + 1
					v = h0 * l0 * w0
					if v < task.volume:
						continue
					start0_location = (h - h0 + 1, depth_out[h, i, j], left[h, i, j])
					end0_location = (h, depth_in[h, i, j], right[h, i, j])
					start_location, end_location = self.get_precise_start_and_end_locations_for_task(task,
					                                                                                 start0_location,
					                                                                                 end0_location)
					if not is_back_filling_task:
						is_find = flag = True
						break
					else:
						# 处理是back_filling TASK的情况
						if self.check_if_back_filling_task_can_put(start0_location, end0_location, preserve_locations,
						                                           task, waiting_time):
							is_find = flag = True
							break
				if flag:
					break
			if flag:
				break

		if is_find:
			locations = self.get_all_locations_between_two_points(start_location, end_location)
		return is_find, locations

	# 用next fit策略为一个任务分配Nodes
	def next_fit(self, task, time_matrix, last_location, is_back_filling_task=False, waiting_time=0,preserve_locations=None):
		height, depth_out, depth_in, left, right = self.get_auxiliary_data(time_matrix)
		is_find, locations = False, []
		(talls, rows, columns) = self.hpc.shape
		flag = False
		start_location = end_location = (0, 0, 0)
		(last_h, last_i, last_j) = last_location
		for h in range(last_h, talls):
			for i in range(last_i, rows):
				for j in range(last_j, columns):
					h0 = height[h, i, j]
					l0 = right[h, i, j] - left[h, i, j] + 1
					w0 = depth_in[h, i, j] - depth_out[h, i, j] + 1
					v = h0 * l0 * w0
					if v < task.volume:
						continue
					start0_location = (h - h0 + 1, depth_out[h, i, j], left[h, i, j])
					end0_location = (h, depth_in[h, i, j], right[h, i, j])
					start_location, end_location = self.get_precise_start_and_end_locations_for_task(task,
					                                                                                 start0_location,
					                                                                                 end0_location)
					if not is_back_filling_task:
						is_find = flag = True
						break
					else:
						# 处理是back_filling TASK的情况
						if self.check_if_back_filling_task_can_put(start0_location, end0_location, preserve_locations, task, waiting_time):
							is_find = flag = True
							break
				if flag:
					break
			if flag:
				break

		if is_find:
			locations = self.get_all_locations_between_two_points(start_location, end_location)
			return is_find, locations

		# 如果一开始就是从头开始找的，那么就没必要进行后面的first_fit了
		if op.eq(last_location, (0, 0, 0)):
			return is_find, locations

		return self.first_fit(task, time_matrix, is_back_filling_task, waiting_time, preserve_locations)

	# best fit策略为一个任务分配Nodes
	def best_fit(self, task, time_matrix, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		height, depth_out, depth_in, left, right = self.get_auxiliary_data(time_matrix)
		is_find, locations = False, []
		(talls, rows, columns) = self.hpc.shape
		start_location = end_location = (0, 0, 0)
		last_volume = self.total_nodes
		flag = False
		for h in range(talls):
			for i in range(rows):
				for j in range(columns):
					h0 = height[h, i, j]
					l0 = right[h, i, j] - left[h, i, j] + 1
					w0 = depth_in[h, i, j] - depth_out[h, i, j] + 1
					v = h0 * l0 * w0
					if v < task.volume:
						continue
					start0_location = (h - h0 + 1, depth_out[h, i, j], left[h, i, j])
					end0_location = (h, depth_in[h, i, j], right[h, i, j])
					if not is_back_filling_task:
						is_find = True
						cur_volumes = self.get_volumes_between_two_points(start0_location, end0_location)
						if cur_volumes < last_volume:
							last_volume = cur_volumes
							start_location, end_location = self.get_precise_start_and_end_locations_for_task(task,
							                                                                                 start0_location,
							                                                                                 end0_location)
							if cur_volumes == task.volume:
								flag = True
								break
					else:
						# 处理是back_filling TASK的情况
						if self.check_if_back_filling_task_can_put(start0_location, end0_location, preserve_locations,
						                                           task, waiting_time):
							is_find = True
							cur_volumes = self.get_volumes_between_two_points(start0_location, end0_location)
							if cur_volumes < last_volume:
								last_volume = cur_volumes
								start_location, end_location = self.get_precise_start_and_end_locations_for_task(task,
								                                                                                 start0_location,
								                                                                                 end0_location)
								if cur_volumes == task.volume:
									flag = True
									break
				if flag:
					break
			if flag:
				break

		if is_find:
			locations = self.get_all_locations_between_two_points(start_location, end_location)
		return is_find, locations

	def worst_fit(self, task, time_matrix, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		height, depth_out, depth_in, left, right = self.get_auxiliary_data(time_matrix)
		is_find, locations = False, []
		(talls, rows, columns) = self.hpc.shape
		start_location = end_location = (0, 0, 0)
		last_volume = 0
		flag = False
		for h in range(talls):
			for i in range(rows):
				for j in range(columns):
					h0 = height[h, i, j]
					l0 = right[h, i, j] - left[h, i, j] + 1
					w0 = depth_in[h, i, j] - depth_out[h, i, j] + 1
					v = h0 * l0 * w0
					if v < task.volume:
						continue
					start0_location = (h - h0 + 1, depth_out[h, i, j], left[h, i, j])
					end0_location = (h, depth_in[h, i, j], right[h, i, j])
					if not is_back_filling_task:
						is_find = True
						cur_volumes = self.get_volumes_between_two_points(start0_location, end0_location)
						if cur_volumes > last_volume:
							last_volume = cur_volumes
							start_location, end_location = self.get_precise_start_and_end_locations_for_task(task,
							                                                                                 start0_location,
							                                                                                 end0_location)

							if cur_volumes == self.total_nodes:
								flag = True
								break
					else:
						# 处理是back_filling TASK的情况
						if self.check_if_back_filling_task_can_put(start0_location, end0_location, preserve_locations,
						                                           task, waiting_time):
							is_find = True
							cur_volumes = self.get_volumes_between_two_points(start0_location, end0_location)
							if cur_volumes > last_volume:
								last_volume = cur_volumes
								start_location, end_location = self.get_precise_start_and_end_locations_for_task(task,
								                                                                                 start0_location,
								                                                                                 end0_location)

								if cur_volumes == self.total_nodes:
									flag = True
									break
				if flag:
					break
			if flag:
				break

		if is_find:
			locations = self.get_all_locations_between_two_points(start_location, end_location)
		return is_find, locations

	def increment_target_job_num(self, task):
		job_id = int(task.name[1:])
		if 1520 < job_id <= 3530:
			self.target_fit_job_num += 1
			# print(task.name + ' has been scheduled')

	def update_hpc(self, locations, task):
		# self.lock.acquire()
		# try:
		# 因为有可能找到的连续Nodes数量大于task的需求，所以要做判断，满足task需求了即可
		for t in locations:
			self.hpc[t[0], t[1], t[2]] = task.time
			self.job_name_matrix[t[0], t[1], t[2]] = int(task.name[1:])

		self.empty_nodes = self.empty_nodes - len(locations)
		self.has_scheduled_task_num = self.has_scheduled_task_num + 1
		self.increment_target_job_num(task)
		# if self.has_scheduled_task_num % 10 == 0:
		# 	print('{}: {}'.format(self.method_name.value, self.has_scheduled_task_num))

		wasted_locations = locations[task.volume:]
		# print('wasted_nodes: {}'.format(len(wasted_locations)))
		self.wasted_nodes = self.wasted_nodes + len(wasted_locations)
		record = RunningRecord(task.name, len(locations), task.time, wasted_locations)
		self.running_task_list.append(record)
		self.running_task_list.sort(key=lambda x: x.rest_time, reverse=False)

		self.running_task_ids.add(task.name)

	# if self.enable_visualize:
	# 	self.post_process_voxels()

	# finally:
	# 	self.lock.release()

	# def post_process_voxels(self):
	# 	locations = self.get_all_wasted_nodes_locations()
	# 	vMsg = OldVisualizerMsg(VMsgType.CONTINUE, self.hpc, locations)
	# 	self.voxelQueue.put(vMsg)

	def get_all_wasted_nodes_locations(self):
		locations = []
		for r in self.running_task_list:
			for t in r.wasted_nodes_locations:
				locations.append(t)
		return locations

	def universal_find_nodes_and_min_wait_time(self, first_task):
		locations = []
		min_wait_time = sys.maxsize
		tmp = min_wait_time
		will_empty_nodes = 0
		for r in self.running_task_list:
			will_empty_nodes = will_empty_nodes + r.volume
			if will_empty_nodes + self.empty_nodes < first_task.volume:
				continue
			tmp_time_matrix = self.get_possible_time_matrix(np.repeat(self.hpc, 1, axis=0), r.rest_time)
			result, locations = self.schedule(FitMethodType.FIRST_FIT, first_task, tmp_time_matrix)
			if result:
				# print('wait_time: {}\tlocations: {}'.format(r.rest_time, locations))
				# self.test_flag = False
				min_wait_time = r.rest_time
				break
		# if min_wait_time == tmp:
		# 	r = self.running_task_list[-1]
		# 	min_wait_time = r.rest_time
		return locations, min_wait_time

	def get_possible_time_matrix(self, matrix, judge_state):
		[height, rows, columns] = matrix.shape
		for h in range(height):
			for i in range(rows):
				for j in range(columns):
					v = matrix[h, i, j] - judge_state
					if v >= 0:
						matrix[h, i, j] = v
					else:
						matrix[h, i, j] = 0
		return matrix

	def start_offline_back_filling(self, next_index, waiting_time, preserve_locations):
		cur_scheduled_tasks_num = self.has_scheduled_task_num
		for i in range(next_index, len(self.taskQueue)):
			if self.empty_nodes == 0:
				break
			task = self.taskQueue[i]
			if self.has_run(task) or self.empty_nodes < task.volume or task.volume >= 50:
				continue
			result, locations = self.schedule(FitMethodType.FIRST_FIT, task, self.hpc, is_back_filling_task=True,
			                                  waiting_time=waiting_time, preserve_locations=preserve_locations)
			if result:
				self.do_after_find(task, locations)
		# 如果不等，说明有任务放上去了，这次bf是成功的
		return cur_scheduled_tasks_num != self.has_scheduled_task_num

	def start_online_back_filling(self, waiting_time, preserve_locations):
		cur_scheduled_tasks_num = self.has_scheduled_task_num
		start_index = self.poisson_queue_cursor
		end_index = len(self.poisson_task_queue)
		can_bf_task_num = self.arrival_rate
		task_num_counter = 0
		for i in range(start_index, end_index):
			if self.empty_nodes == 0:
				break
			task = self.poisson_task_queue[i]
			if self.has_run(task) or self.empty_nodes < task.volume or task.volume >= 50:
				continue
			result, locations = self.schedule(FitMethodType.FIRST_FIT, task, self.hpc, is_back_filling_task=True, waiting_time=waiting_time, preserve_locations=preserve_locations)
			if result:
				self.do_after_find(task, locations)
			task_num_counter += 1
			if self.time_strategy == 2 and task_num_counter >= can_bf_task_num:
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
				break
		# 如果不等，说明有任务放上去了，这次bf是成功的
		return cur_scheduled_tasks_num != self.has_scheduled_task_num

	def start_online_back_filling_with_sort(self, waiting_time, preserve_locations):
		cur_scheduled_tasks_num = self.has_scheduled_task_num
		can_bf_task_num = self.arrival_rate
		task_num_counter = 0
		false_task_list = []
		while self.sorted_queue:
			if self.empty_nodes == 0:
				break
			task = heapq.heappop(self.sorted_queue)
			if self.has_run(task):
				continue
			if self.empty_nodes < task.volume or task.volume >= 50:
				false_task_list.append(task)
				continue
			result, locations = self.schedule(FitMethodType.FIRST_FIT, task, self.hpc, is_back_filling_task=True, waiting_time=waiting_time, preserve_locations=preserve_locations)
			if result:
				self.do_after_find(task, locations)
			else:
				false_task_list.append(task)
			task_num_counter += 1
			if self.time_strategy == 2 and task_num_counter >= can_bf_task_num:
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
				self.move_task_from_prototype_to_sorted_queue()
				break
		for t in false_task_list:
			heapq.heappush(self.sorted_queue, t)
		# 如果不等，说明有任务放上去了，这次bf是成功的
		return cur_scheduled_tasks_num != self.has_scheduled_task_num

	def count_utilization(self, counter):
		if not self.start_record_util:
			return
		using_nodes_num = self.total_nodes - self.empty_nodes - self.wasted_nodes
		cur_util = using_nodes_num * 1.0 / self.total_nodes
		if cur_util == 0.0:
			return
		str_cur_util = str(cur_util)
		if counter % 5 == 0:
			self.u5.append(str_cur_util)
		if counter % 10 == 0:
			self.u10.append(str_cur_util)
		if counter % 15 == 0:
			self.u15.append(str_cur_util)

	def process_can_not_schedule(self, task, task_index, is_online=False):
		locations, wait_time = self.universal_find_nodes_and_min_wait_time(task)
		# print('{} 需要等待{}, 需要{}个nodes'.format(task.name, wait_time, len(locations)))
		# 在这一时间立即开始一次back-filling
		# 之后把这里的改成self.has_scheduled_task_num >= 1520，这里为了测试暂时改成0了
		if self.enable_back_filling and self.has_scheduled_task_num >= 1520:
			left_waiting_time = wait_time
			if left_waiting_time > 0:
				if is_online:
					if not self.enable_sort:
						self.start_online_back_filling(left_waiting_time, locations)
					else:
						self.start_online_back_filling_with_sort(left_waiting_time, locations)
				else:
					self.start_offline_back_filling(task_index + 1, left_waiting_time, locations)
			self.count_utilization(self.time_counter)
		cur_empty_nodes = self.empty_nodes
		last_bf_success = True
		for i in range(wait_time - 1, -1, -1):
			if self.has_scheduled_task_num >= 1520:
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
			self.time_process()
			if is_online and self.has_left_task_to_fetch and self.has_scheduled_task_num >= 1520:
				if self.enable_sort:
					move_task_num = self.move_task_from_prototype_to_sorted_queue()
				else:
					move_task_num = self.move_task_from_prototype_to_poisson()
				if move_task_num == 0:
					self.has_left_task_to_fetch = False
			# 之后把这里的改成self.has_scheduled_task_num >= 1520，这里为了测试暂时改成0了
			if self.enable_back_filling and self.has_scheduled_task_num >= 1520:
				left_waiting_time = i
				empty_nodes_num_changed = (cur_empty_nodes != self.empty_nodes)
				if empty_nodes_num_changed:
					cur_empty_nodes = self.empty_nodes
				# 如果经过一个单位时间之后空nodes数量没变且上一次bf没有任务被调度上去，那么这一次也不会有任务被调度上去
				if left_waiting_time > 0:
					if is_online:
						if not self.enable_sort:
							self.start_online_back_filling(left_waiting_time, locations)
						else:
							self.start_online_back_filling_with_sort(left_waiting_time, locations)
					elif empty_nodes_num_changed or last_bf_success:
						last_bf_success = self.start_offline_back_filling(task_index + 1, left_waiting_time, locations)
		return locations

	def offline_simulate(self):
		# self.generate_offline_task()
		# print(len(self.taskQueue))
		last_location = (0, 0, 0)
		locations = []
		for i, task in enumerate(self.taskQueue):
			if self.can_program_stop:
				break
			if self.has_run(task):
				continue
			found, locations = self.schedule(self.method_name, task, time_matrix=self.hpc, last_location=last_location)
			if not found:
				locations = self.process_can_not_schedule(task, i)
			self.do_after_find(task, locations)
			# print('task:{}'.format(task))
			# print('{}:{}'.format(found, locations))
			last_location = locations[task.volume - 1]
			locations.clear()

	def online_simulate(self):
		last_location = (0, 0, 0)
		locations = []
		task_num = len(self.prototype_queue)
		schedule_task_num_here = 0
		while self.has_scheduled_task_num < task_num and not self.can_program_stop:
			if self.poisson_queue_cursor >= len(self.poisson_task_queue):
				move_task_num = self.move_task_from_prototype_to_poisson()
				if move_task_num == 0:
					self.has_left_task_to_fetch = False
					break
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)

			task = self.poisson_task_queue[self.poisson_queue_cursor]
			self.poisson_queue_cursor = self.poisson_queue_cursor + 1
			if self.has_run(task):
				continue

			# 大于50后面改回来
			# if task.volume >= 50:
			# 	self.running_task_ids.add(task.name)
			# 	self.increment_target_job_num(task)
			# 	continue

			found, locations = self.schedule(self.method_name, task, time_matrix=self.hpc, last_location=last_location)
			schedule_task_num_here += 1
			if self.time_strategy == 2 and schedule_task_num_here == self.arrival_rate:
				# 从头开始计数
				schedule_task_num_here = 0
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
			if not found:
				locations = self.process_can_not_schedule(task, -1, is_online=True)
			self.do_after_find(task, locations)
			last_location = locations[task.volume - 1]
			locations.clear()

	def move_task_from_prototype_to_poisson(self):
		move_task_num = self.arrival_rate
		left_task_num = len(self.prototype_queue) - self.prototype_queue_cursor
		if left_task_num == 0:
			return
		if left_task_num < move_task_num:
			move_task_num = left_task_num
		for i in range(self.prototype_queue_cursor, self.prototype_queue_cursor + move_task_num):
			task = self.prototype_queue[i]
			self.poisson_task_queue.append(task)
		self.prototype_queue_cursor = self.prototype_queue_cursor + move_task_num
		return move_task_num

	def online_simulate_with_sort_time(self):
		last_location = (0, 0, 0)
		locations = []
		task_num = len(self.prototype_queue)
		schedule_task_num_here = 0
		while self.has_scheduled_task_num < task_num and not self.can_program_stop:
			if self.poisson_queue_cursor < 1520:
				task = self.poisson_task_queue[self.poisson_queue_cursor]
				self.poisson_queue_cursor = self.poisson_queue_cursor + 1
			else:
				if len(self.sorted_queue) == 0:
					move_task_num = self.move_task_from_prototype_to_sorted_queue()

					if move_task_num == 0:
						self.has_left_task_to_fetch = False
						break
					self.time_process()
					self.time_counter = self.time_counter + 1
					self.count_utilization(self.time_counter)
				task = heapq.heappop(self.sorted_queue)

			if self.has_run(task):
				continue

			# 大于50后面改回来
			# if task.volume >= 50:
			# 	self.running_task_ids.add(task.name)
			# 	self.increment_target_job_num(task)
			# 	continue

			found, locations = self.schedule(self.method_name, task, time_matrix=self.hpc, last_location=last_location)
			if self.has_scheduled_task_num >= 1520:
				schedule_task_num_here += 1
			if self.time_strategy == 2 and schedule_task_num_here == self.arrival_rate:
				# 从头开始计数
				schedule_task_num_here = 0
				self.move_task_from_prototype_to_sorted_queue()
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
			if not found:
				locations = self.process_can_not_schedule(task, -1, is_online=True)
			self.do_after_find(task, locations)
			last_location = locations[task.volume - 1]
			locations.clear()

	def move_task_from_prototype_to_sorted_queue(self):
		move_task_num = self.arrival_rate
		left_task_num = len(self.prototype_queue) - self.prototype_queue_cursor
		if left_task_num == 0:
			return 0
		if left_task_num < move_task_num:
			move_task_num = left_task_num
		for i in range(self.prototype_queue_cursor, self.prototype_queue_cursor + move_task_num):
			task = self.prototype_queue[i]
			heapq.heappush(self.sorted_queue, task)
		self.prototype_queue_cursor = self.prototype_queue_cursor + move_task_num
		return move_task_num


if __name__ == '__main__':
	v = 4
	hpc_size = [v, v, v]
	# scheduler3D = OldTaskSchedulerFor3D(hpc_size, [], method_name=FitMethodType.NEXT_FIT, enable_back_filling=True)
	# scheduler3D.generate_task()
	# start_time = time.time()
	# scheduler3D.online_simulate()
	# end_time = time.time()
	# print('{}秒'.format(end_time-start_time))
	# print(scheduler3D.u5)
