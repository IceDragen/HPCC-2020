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

from Simulator.Entities import FitMethodType, Task, RunningRecord


class Conventional:
	# hpc: the HPC system
	# size: the size of HPC system
	# data_path: the file path to store the utilization data
	# time_path: the file path to store the cost time of a simulation
	# arrival_rate: the lambda of Poisson process
	def __init__(self, size, task_queue, data_path=None, time_path=None, arrival_rate=2,
	             method_name=FitMethodType.FIRST_FIT,
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

		self.prototype_queue = self.taskQueue
		self.poisson_task_queue = self.taskQueue[:1520]
		self.has_left_task_to_fetch = True
		self.time_strategy = st
		self.prototype_queue_cursor = 1520
		self.poisson_queue_cursor = 0
		self.arrival_rate = arrival_rate

		self.sorted_queue = []

		self.counter = 0
		self.last_location = (0, 0, 0)

		self.visualizer = None

	def job_name_matrix_init(self):
		[height, rows, columns] = self.job_name_matrix.shape
		for i in range(rows):
			for j in range(columns):
				for h in range(height):
					self.job_name_matrix[i, j] = 0

	def generate_task(self):
		i = 1
		for t in range(20):
			j_name = 'j' + str(i)
			volume = random.randint(1, int(self.total_nodes / 2))
			cost_time = random.randint(1, 10)
			task = Task(j_name, volume, cost_time)
			self.taskQueue.append(task)
			i = i + 1
		self.poisson_task_queue = self.taskQueue[:5]
		self.prototype_queue_cursor = 5
		self.poisson_queue_cursor = 0

	# judge if a task has been scheduled
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

	def schedule(self, method_name, task, time_matrix, last_location=None, is_back_filling_task=False, waiting_time=0,
	             preserve_locations=None):
		is_find = False
		locations = []

		if method_name == FitMethodType.FIRST_FIT:
			is_find, locations = self.first_fit(task, time_matrix, is_back_filling_task, waiting_time,
			                                    preserve_locations)
		elif method_name == FitMethodType.BEST_FIT:
			is_find, locations = self.best_fit(task, time_matrix, is_back_filling_task, waiting_time,
			                                   preserve_locations)
		elif method_name == FitMethodType.WORST_FIT:
			is_find, locations = self.worst_fit(task, time_matrix, is_back_filling_task, waiting_time,
			                                    preserve_locations)
		elif method_name == FitMethodType.NEXT_FIT:
			is_find, locations = self.next_fit(task, time_matrix, last_location, is_back_filling_task, waiting_time,
			                                   preserve_locations)

		return is_find, locations

	# the runnning jobs' rest time minus one minute
	def time_process(self):
		[height, rows, columns] = self.hpc.shape
		for h in range(height):
			for i in range(rows):
				for j in range(columns):
					if self.hpc[h, i, j] > 0:
						self.hpc[h, i, j] = self.hpc[h, i, j] - 1
					if self.hpc[h, i, j] == 0:
						self.job_name_matrix[h, i, j] = 0

		if self.enable_back_filling:
			if self.counter > 0:
				self.counter = self.counter - 1

		self.check_and_update_record()

	# remove record from self.running_task_records which records'rest time is 0
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

	# judge if start or end recording system utilization
	def check_if_adjust_record_util(self):
		if 'j3530' in self.running_task_ids and self.target_fit_job_num == 2010:
			self.start_record_util = False
			if not self.has_record_data:
				self.end_time = time.time()
				self.write_utilization_to_file()
				self.write_cost_time_to_file()
				self.has_record_data = True
				self.can_program_stop = True

		if 'j1520' in self.running_task_ids:
			self.start_record_util = True
			if not self.has_begin_time_count:
				self.start_time = time.time()
				self.has_begin_time_count = True

	def write_utilization_to_file(self):
		delimiter = ','
		data5 = delimiter.join(self.u5)
		all_data = [data5]
		with open(self.util_data_file_path, 'a+') as f:
			for data in all_data:
				f.write(data)
				f.write('\n')

	def write_cost_time_to_file(self):
		cost_time = self.end_time - self.start_time
		with open(self.time_data_file_path, 'a+') as f:
			f.write(str(cost_time))
			f.write('\n')

	# process the task that can be scheduled currently
	def do_after_find(self, task, locations):
		self.update_hpc(locations, task)
		self.check_if_adjust_record_util()

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

	# get the four auxiliary matrix for finding a cuboid to put the task
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
					if matrix[h, i, j] == 0:
						if h - 1 < 0:
							height[h, i, j] = 1
						else:
							height[h, i, j] = height[h - 1, i, j] + 1
					else:
						height[h, i, j] = 0

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

	# calculate the amount of nodes between given two locations
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

	# the found cuboid maybe larger then the nodes of the task requested, so we must get the presize locations to put the task
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
			return start_location, over_location
		elif s >= task.volume:
			tmp = math.ceil(task.volume / width)
			return start_location, (start_h, start_i + tmp - 1, end_j)
		else:
			tmp = math.ceil(task.volume / s)
			return start_location, (start_h + tmp - 1, end_i, end_j)

	# judge if test_location is in the middle of start_location and end_location
	def judge_in_middle(self, start_location, end_location, judge_location):
		(start_h, start_i, start_j) = start_location
		(end_h, end_i, end_j) = end_location
		(j_h, j_i, j_j) = judge_location
		a = start_h <= j_h <= end_h
		b = start_i <= j_i <= end_i
		c = start_j <= j_j <= end_j
		return a and b and c

	# judge if the task can be put in between the given two nodes
	def check_if_back_filling_task_can_put(self, start_location, end_location, preserve_locations, task, waiting_time):
		r1 = self.judge_in_middle(preserve_locations[0], preserve_locations[-1], start_location)
		r2 = self.judge_in_middle(preserve_locations[0], preserve_locations[-1], end_location)
		if r1 or r2:
			if task.time <= waiting_time:
				return True
			else:
				return False
		return True

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

	def next_fit(self, task, time_matrix, last_location, is_back_filling_task=False, waiting_time=0,
	             preserve_locations=None):
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

		if op.eq(last_location, (0, 0, 0)):
			return is_find, locations

		return self.first_fit(task, time_matrix, is_back_filling_task, waiting_time, preserve_locations)

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

	# judge if current scheduled task is belong to the first synthetic workload
	def increment_target_job_num(self, task):
		job_id = int(task.name[1:])
		if 1520 < job_id <= 3530:
			self.target_fit_job_num += 1

	# update self.hpc, update the task rest-time to the corresponding locations
	def update_hpc(self, locations, task):
		for t in locations:
			self.hpc[t[0], t[1], t[2]] = task.time
			self.job_name_matrix[t[0], t[1], t[2]] = int(task.name[1:])

		self.empty_nodes = self.empty_nodes - len(locations)
		self.has_scheduled_task_num = self.has_scheduled_task_num + 1
		self.increment_target_job_num(task)

		wasted_locations = locations[task.volume:]
		self.wasted_nodes = self.wasted_nodes + len(wasted_locations)
		record = RunningRecord(task.name, len(locations), task.time, wasted_locations)
		self.running_task_list.append(record)
		self.running_task_list.sort(key=lambda x: x.rest_time, reverse=False)

		self.running_task_ids.add(task.name)

	def get_all_wasted_nodes_locations(self):
		locations = []
		for r in self.running_task_list:
			for t in r.wasted_nodes_locations:
				locations.append(t)
		return locations

	# get the locations and waiting time for current blocked job
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
				min_wait_time = r.rest_time
				break
		return locations, min_wait_time

	# a auxiliary function to find after how many minutes the blocked task can be put on the system
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

	# start backfilling in offline situation
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
		return cur_scheduled_tasks_num != self.has_scheduled_task_num

	# start backfilling in online situation with FCFS strategy
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
			result, locations = self.schedule(FitMethodType.BEST_FIT, task, self.hpc, is_back_filling_task=True,
			                                  waiting_time=waiting_time, preserve_locations=preserve_locations)
			if result:
				self.do_after_find(task, locations)
			task_num_counter += 1
			if self.time_strategy == 2 and task_num_counter >= can_bf_task_num:
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
				break
		return cur_scheduled_tasks_num != self.has_scheduled_task_num

	# start backfilling in online situation with SJF strategy
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
			result, locations = self.schedule(FitMethodType.BEST_FIT, task, self.hpc, is_back_filling_task=True,
			                                  waiting_time=waiting_time, preserve_locations=preserve_locations)
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
		return cur_scheduled_tasks_num != self.has_scheduled_task_num

	# count current system utilization
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

	# process the task cannot scheduled currently
	def process_can_not_schedule(self, task, task_index, is_online=False):
		locations, wait_time = self.universal_find_nodes_and_min_wait_time(task)
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
			if self.enable_back_filling and self.has_scheduled_task_num >= 1520:
				left_waiting_time = i
				empty_nodes_num_changed = (cur_empty_nodes != self.empty_nodes)
				if empty_nodes_num_changed:
					cur_empty_nodes = self.empty_nodes
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
			last_location = locations[task.volume - 1]
			locations.clear()

	def online_simulate_with_FCFS(self):
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

			found, locations = self.schedule(self.method_name, task, time_matrix=self.hpc, last_location=last_location)
			schedule_task_num_here += 1
			if self.time_strategy == 2 and schedule_task_num_here == self.arrival_rate:
				schedule_task_num_here = 0
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
			if not found:
				locations = self.process_can_not_schedule(task, -1, is_online=True)
			self.do_after_find(task, locations)
			last_location = locations[task.volume - 1]
			locations.clear()

	# move λ tasks from prototype queue to the task queue which is a list for FCFS
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

	def online_simulate_with_SJF(self):
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

			found, locations = self.schedule(self.method_name, task, time_matrix=self.hpc, last_location=last_location)
			if self.has_scheduled_task_num >= 1520:
				schedule_task_num_here += 1
			if self.time_strategy == 2 and schedule_task_num_here == self.arrival_rate:
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

	# move λ tasks from prototype queue to the task queue which is a priority queue for SJF
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
