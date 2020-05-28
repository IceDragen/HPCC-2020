import os
import random
import sys
import time

from queue import Queue, PriorityQueue
import heapq

import numpy as np

from Simulator.Entities import FitMethodType, Task, RunningRecord


class Wireless:
	# hpc: the HPC system
	# size: the size of HPC system
	# data_path: the file path to store the utilization data
	# time_path: the file path to store the cost time of a simulation
	# arrival_rate: the lambda of Poisson process
	def __init__(self, size, task_queue, arrival_rate=200, method_name=FitMethodType.FIRST_FIT,
	             data_path=None, time_path=None, enable_back_filling=False, enable_visualize=False, st=1,
	             enable_time_sort=False):
		self.enable_back_filling = enable_back_filling
		self.taskQueue = task_queue
		self.run_out_task_num = 0
		self.has_scheduled_task_num = 0
		self.util_data_file_path = data_path
		self.time_data_file_path = time_path
		self.enable_sort = enable_time_sort

		self.visualQueue = Queue(maxsize=0)
		self.first_display = True
		self.enable_visualize = enable_visualize
		self.visualizer = None

		self.hpc = np.zeros(size, dtype=int)
		self.job_name_matrix = np.zeros(size, dtype=int)
		self.total_nodes = size[0] * size[1] * size[2]
		self.empty_nodes = self.total_nodes

		self.method_name = method_name
		self.target_fit_job_num = 0
		self.can_program_stop = False

		self.running_task_records = []
		self.running_task_ids = set()

		self.time_counter = 0
		self.start_record_util = False
		self.start_time = None
		self.end_time = None
		self.has_start_time_count = False
		self.has_record_data = False

		self.u5 = []
		self.u10 = []
		self.u15 = []

		self.time_strategy = st
		self.prototype_queue = self.taskQueue
		self.poisson_task_queue = self.taskQueue[:1520]
		self.has_left_task_to_fetch = True
		self.prototype_queue_cursor = 1520
		self.poisson_queue_cursor = 0
		self.arrival_rate = arrival_rate

		self.sorted_queue = []

		self.counter = 0
		self.last_location_for_back_fill = (-1, 0, 0)

		self.test_flag = True
		self.not_zero_counter = 0

	# judge if a task has been scheduled
	def has_run(self, task):
		return task.name in self.running_task_ids

	def create_columns_list(self):
		columns_list = []
		[_, _, columns] = self.hpc.shape
		for i in range(columns):
			columns_list.append(i)
		return columns_list

	def print_time_matrix(self):
		[height, rows, columns] = self.hpc.shape
		columns_list = self.create_columns_list()
		for h in range(height - 1, -1, -1):
			if columns_list[0] != 0:
				columns_list.reverse()
			for i in range(rows):
				for j in columns_list:
					print('%4d' % self.hpc[h, i, j], end=' ')
				print(end='\t')
				columns_list.reverse()
			print()

	def print_job_matrix(self):
		[height, rows, columns] = self.hpc.shape
		columns_list = self.create_columns_list()
		for h in range(height - 1, -1, -1):
			if columns_list[0] != 0:
				columns_list.reverse()
			for i in range(rows):
				for j in columns_list:
					print('%4d' % self.job_name_matrix[h, i, j], end=' ')
				print(end='\t')
				columns_list.reverse()
			print()

	def schedule(self, task, last_location, is_back_filling_task=False, waiting_time=0,
	             preserve_locations=None):
		found = False
		locations = []

		# print(task)
		if self.method_name == FitMethodType.FIRST_FIT:
			found, locations = self.first_fit(task, 0, is_back_filling_task, waiting_time, preserve_locations)
		elif self.method_name == FitMethodType.BEST_FIT:
			found, locations = self.best_fit(task, 0, is_back_filling_task, waiting_time, preserve_locations)
		elif self.method_name == FitMethodType.WORST_FIT:
			found, locations = self.worst_fit(task, 0, is_back_filling_task, waiting_time, preserve_locations)
		elif self.method_name == FitMethodType.NEXT_FIT:
			found, locations = self.next_fit(task, last_location, 0, is_back_filling_task, waiting_time,
			                                 preserve_locations)

		return found, locations

	# the runnning jobs' rest time minus one minute
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
						if self.enable_visualize and self.hpc[h, i, j] == 0:
							need_visualize_change = True
					if self.hpc[h, i, j] == 0:
						self.job_name_matrix[h, i, j] = 0

		self.check_and_update_record()

		if self.counter > 0:
			self.counter = self.counter - 1

	# remove record from self.running_task_records which records'rest time is 0
	def check_and_update_record(self):
		removing_id_list = []
		for i, r in enumerate(self.running_task_records):
			if r.rest_time - 1 <= 0:
				removing_id_list.append(i)
				self.empty_nodes = self.empty_nodes + r.volume
			else:
				r.rest_time = r.rest_time - 1

		self.run_out_task_num = self.run_out_task_num + len(removing_id_list)

		if len(removing_id_list) > 0:
			new_running_task = [t for i, t in enumerate(self.running_task_records) if i not in removing_id_list]
			self.running_task_records = new_running_task

	def write_utilization_to_file(self):
		delimiter = ','
		data5 = delimiter.join(self.u5)
		all_data = [data5]

		with open(self.util_data_file_path, 'a+') as f:
			for data in all_data:
				f.write(data + '\n')

	def write_cost_time_to_file(self):
		cost_time = self.end_time - self.start_time
		with open(self.time_data_file_path, 'a+') as f:
			f.write(str(cost_time))
			f.write('\n')

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
			if not self.has_start_time_count:
				self.start_time = time.time()
				self.has_start_time_count = True

	# process the task that can be scheduled currently
	def do_after_find(self, task, locations):
		self.update_hpc(locations, task)
		self.check_if_adjust_record_util()


	def get_start_and_end_location(self, locations):
		if locations is not None:
			return locations[0], locations[-1]
		return ()

	def first_fit(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		count = 0
		locations = []
		columns_list = self.create_columns_list()
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)

		for i in range(rows):
			for j in columns_list:
				for h in range(height):
					if count >= task.volume:
						break
					if self.hpc[h, i, j] <= judge_state:
						if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
						                                                 end_location) and task.time > waiting_time:
							count = 0
							locations.clear()
							continue
						count = count + 1
						locations.append((h, i, j))
					elif self.hpc[h, i, j] > judge_state:
						count = 0
						locations.clear()
				if count >= task.volume:
					break
			if count >= task.volume:
				break
			columns_list.reverse()

		if count < task.volume:
			return False, []

		return True, locations

	def next_fit(self, task, last_location, judge_state=0, is_back_filling_task=False, waiting_time=0,
	             preserve_locations=None):

		[height, rows, columns] = self.hpc.shape
		count = 0
		locations = []

		found, next_location = self.next_location(last_location)

		if not found:
			result, locations = self.sub_process(task, judge_state, is_back_filling_task, waiting_time,
			                                     preserve_locations)
			if result:
				return result, locations
			else:
				return False, []

		(start_h, start_i, start_j) = next_location
		if start_i % 2 == 0:
			columns_list = [x for x in range(start_j, columns)]
		else:
			columns_list = [x for x in range(start_j, -1, -1)]
		height_list = [x for x in range(start_h, height)]
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)

		for i in range(start_i, rows):
			for j in columns_list:
				for h in height_list:
					if count >= task.volume:
						break
					if self.hpc[h, i, j] <= judge_state:
						if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
						                                                 end_location) and task.time > waiting_time:
							count = 0
							locations.clear()
							continue
						count = count + 1
						locations.append((h, i, j))
					elif self.hpc[h, i, j] > judge_state:
						count = 0
						locations.clear()
				if count >= task.volume:
					break
				if len(height_list) < height:
					for value in range(start_h):
						height_list.insert(value, value)
			if count >= task.volume:
				break
			if len(columns_list) < columns:
				if start_i % 2 == 0:
					for value in range(start_j):
						columns_list.insert(value, value)
				else:
					columns_list.reverse()
					for value in range(start_j + 1, columns):
						columns_list.append(value)
					columns_list.reverse()
			columns_list.reverse()
		if count < task.volume:
			result, locations = self.sub_process(task, judge_state, is_back_filling_task, waiting_time,
			                                     preserve_locations)
			if result:
				return result, locations
			else:
				return False, []
		else:
			return True, locations

	# a subcall in next_fit
	def sub_process(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		count = 0
		locations = []
		columns_list = self.create_columns_list()
		break_flag = False
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)

		for i in range(rows):
			for j in columns_list:
				for h in range(height):
					if count >= task.volume:
						break_flag = True
						break
					if self.hpc[h, i, j] <= judge_state:
						if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
						                                                 end_location) and task.time > waiting_time:
							count = 0
							locations.clear()
							continue
						count = count + 1
						locations.append((h, i, j))
					elif self.hpc[h, i, j] > judge_state:
						count = 0
						locations.clear()
				if break_flag:
					break
			if break_flag:
				break
			columns_list.reverse()

		if count < task.volume:
			return False, ()
		else:
			return True, locations

	def best_fit(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		old_count = sys.maxsize
		old_locations = []
		cur_count = 0
		locations = []
		flag = False
		columns_list = self.create_columns_list()
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)

		for i in range(rows):
			for j in columns_list:
				for h in range(height):
					if self.hpc[h, i, j] <= judge_state:
						if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
						                                                 end_location) and task.time > waiting_time:
							if cur_count == task.volume:
								old_count = cur_count
								old_locations = locations[:]
								flag = True
								break
							elif task.volume < cur_count < old_count:
								old_count = cur_count
								cur_count = 0
								old_locations = locations[:]
								locations.clear()
							else:
								cur_count = 0
								locations.clear()
							continue
						cur_count = cur_count + 1
						locations.append((h, i, j))
					elif self.hpc[h, i, j] > judge_state:
						if cur_count == task.volume:
							old_count = cur_count
							old_locations = locations[:]
							flag = True
							break
						elif task.volume < cur_count < old_count:
							old_count = cur_count
							cur_count = 0
							old_locations = locations[:]
							locations.clear()
						else:
							cur_count = 0
							locations.clear()
				if flag:
					break
			columns_list.reverse()
			if flag:
				break
		if task.volume <= cur_count < old_count:
			old_count = cur_count
			cur_count = 0
			old_locations = locations[:]
			locations.clear()

		found = False
		if flag:
			found = True
		elif cur_count == self.total_nodes:
			found = True
			old_count = cur_count
			old_locations = locations[:]
		elif old_count != sys.maxsize:
			found = True
		if not found:
			return found, []

		return found, old_locations[:task.volume]

	def worst_fit(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		old_count = -1
		old_locations = []
		cur_count = 0
		locations = []
		flag = False
		columns_list = self.create_columns_list()
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)

		for i in range(rows):
			for j in columns_list:
				for h in range(height):
					if self.hpc[h, i, j] <= judge_state:
						if self.hpc[h, i, j] <= judge_state:
							if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
							                                                 end_location) and task.time > waiting_time:
								if cur_count >= task.volume and cur_count > old_count:
									old_count = cur_count
									old_locations = locations[:]
									cur_count = 0
									locations.clear()
								else:
									cur_count = 0
									locations.clear()
								continue
						cur_count = cur_count + 1
						locations.append((h, i, j))
					if self.hpc[h, i, j] > judge_state:
						if cur_count >= task.volume and cur_count > old_count:
							old_count = cur_count
							old_locations = locations[:]
							cur_count = 0
							locations.clear()
						else:
							cur_count = 0
							locations.clear()
			columns_list.reverse()

		if cur_count >= task.volume and cur_count > old_count:
			old_count = cur_count
			old_locations = locations[:]
			cur_count = 0
			locations.clear()

		found = False
		if old_count >= task.volume:
			found = True

		if not found:
			return found, []
		return found, old_locations[:task.volume]

	# judge if current scheduled task is belong to the first synthetic workload
	def increment_target_job_num(self, task):
		job_id = int(task.name[1:])
		if 1520 < job_id <= 3530:
			self.target_fit_job_num += 1

	# update self.hpc, update the task rest-time to the corresponding locations
	def update_hpc(self, locations, task):
		for t in locations:
			if self.hpc[t[0], t[1], t[2]] != 0:
				self.not_zero_counter += 1
			self.hpc[t[0], t[1], t[2]] = task.time
			self.job_name_matrix[t[0], t[1], t[2]] = int(task.name[1:])

		self.empty_nodes = self.empty_nodes - len(locations)
		self.has_scheduled_task_num = self.has_scheduled_task_num + 1
		self.increment_target_job_num(task)

		record = RunningRecord(task.name, len(locations), task.time)
		self.running_task_records.append(record)
		self.running_task_records.sort(key=lambda x: x.rest_time, reverse=False)

		self.running_task_ids.add(task.name)

		return locations[task.volume - 1]

	# judge if test_location is in the middle of start_location and end_location
	def judge_in_middle(self, test_location, start_location, end_location):
		[cur_height, cur_row, cur_col] = test_location
		[start_height, start_row, start_col] = start_location
		[end_height, end_row, end_col] = end_location

		in_middle = False
		if start_row < cur_row < end_row:
			in_middle = True
		elif cur_row == start_row:
			if cur_col == start_col and cur_height >= start_height:
				in_middle = True
			elif cur_col < start_col and cur_row % 2 == 1:
				in_middle = True
			elif cur_col > start_col and cur_row % 2 == 0:
				in_middle = True
		elif cur_row == end_row:
			if cur_col == end_col and cur_height <= end_height:
				in_middle = True
			elif cur_col < end_col and cur_row % 2 == 0:
				in_middle = True
			elif cur_col > end_col and cur_row % 2 == 1:
				in_middle = True
		return in_middle

	# get the next location of given location
	def next_location(self, location):
		[cur_height, cur_row, cur_col] = location
		[height, row, columns] = self.hpc.shape

		if cur_height + 1 >= height:
			start_h = 0
			start_col = cur_col + 1 if cur_row % 2 == 0 else cur_col - 1
			if start_col >= columns or start_col < 0:
				start_row = cur_row + 1
				start_col = 0 if start_col < 0 else columns - 1
			else:
				start_row = cur_row
		else:
			start_h = cur_height + 1
			start_col = cur_col
			start_row = cur_row

		if start_row < row:
			return True, (start_h, start_row, start_col)
		return False, ()

	# get the locations and waiting time for current blocked job
	def universal_find_nodes_and_min_wait_time(self, first_task):
		locations = []
		min_wait_time = sys.maxsize
		will_empty_nodes = 0
		for r in self.running_task_records:
			will_empty_nodes = will_empty_nodes + r.volume
			if will_empty_nodes + self.empty_nodes < first_task.volume:
				continue
			result, locations = self.first_fit(first_task, judge_state=r.rest_time)
			if result:
				min_wait_time = r.rest_time
				break
		return locations, min_wait_time

	# start backfilling in offline situation
	def start_offline_back_filling(self, next_index, waiting_time, preserve_locations):
		cur_scheduled_task_num = self.has_scheduled_task_num
		for i in range(next_index, len(self.taskQueue)):
			if self.empty_nodes == 0:
				break
			task = self.taskQueue[i]
			if self.has_run(task) or self.empty_nodes < task.volume or task.volume >= 50:
				continue
			result, locations = self.first_fit(task, 0, True, waiting_time, preserve_locations)
			if result:
				self.do_after_find(task, locations)
		return cur_scheduled_task_num != self.has_scheduled_task_num

	# start backfilling in online situation with FCFS strategy
	def start_online_back_filling_with_FCFS(self, waiting_time, preserve_locations):
		cur_scheduled_task_num = self.has_scheduled_task_num
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
			result, locations = self.best_fit(task, 0, True, waiting_time, preserve_locations)
			if result:
				self.do_after_find(task, locations)
			task_num_counter += 1
			if self.time_strategy == 2 and task_num_counter >= can_bf_task_num:
				self.time_process()
				self.time_counter = self.time_counter + 1
				self.count_utilization(self.time_counter)
				break

		return cur_scheduled_task_num != self.has_scheduled_task_num

	# start backfilling in online situation with SJF strategy
	def start_online_back_filling_with_SJF(self, waiting_time, preserve_locations):
		cur_scheduled_task_num = self.has_scheduled_task_num
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
			result, locations = self.best_fit(task, 0, True, waiting_time, preserve_locations)
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

		return cur_scheduled_task_num != self.has_scheduled_task_num

	# count current system utilization
	def count_utilization(self, counter):
		if not self.start_record_util:
			return
		using_nodes_num = self.total_nodes - self.empty_nodes
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
						self.start_online_back_filling_with_FCFS(left_waiting_time, locations)
					else:
						self.start_online_back_filling_with_SJF(left_waiting_time, locations)
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
				empty_nodes_changed = (cur_empty_nodes != self.empty_nodes)
				if left_waiting_time > 0:
					if is_online:
						if not self.enable_sort:
							self.start_online_back_filling_with_FCFS(left_waiting_time, locations)
						else:
							self.start_online_back_filling_with_SJF(left_waiting_time, locations)
					elif empty_nodes_changed or last_bf_success:
						last_bf_success = self.start_offline_back_filling(task_index + 1, left_waiting_time, locations)
		return locations

	def offline_simulate(self):
		last_location = (-1, 0, 0)
		locations = []
		for i, task in enumerate(self.taskQueue):
			if self.can_program_stop:
				break
			if self.has_run(task):
				continue
			found, locations = self.schedule(task, last_location)
			if not found:
				locations = self.process_can_not_schedule(task, i)
			self.do_after_find(task, locations)
			last_location = locations[task.volume - 1]
			locations.clear()

	def online_simulate_with_FCFS(self):
		last_location = (-1, 0, 0)
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
			found, locations = self.schedule(task, last_location)
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
			found = False

	# move λ tasks from prototype queue to the task queue which is a list for FCFS
	def move_task_from_prototype_to_poisson(self):
		move_task_num = self.arrival_rate
		left_task_num = len(self.prototype_queue) - self.prototype_queue_cursor
		if left_task_num == 0:
			return 0
		if left_task_num < move_task_num:
			move_task_num = left_task_num
		for i in range(self.prototype_queue_cursor, self.prototype_queue_cursor + move_task_num):
			task = self.prototype_queue[i]
			self.poisson_task_queue.append(task)
		self.prototype_queue_cursor = self.prototype_queue_cursor + move_task_num
		return move_task_num

	def online_simulate_with_SJF(self):
		last_location = (-1, 0, 0)
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
			found, locations = self.schedule(task, last_location)
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
			found = False

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



