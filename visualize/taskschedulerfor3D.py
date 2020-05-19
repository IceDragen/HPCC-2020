import operator
import random
import sys
import threading
import time
from enum import unique, Enum

import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler

from queue import Queue

from visualize.Visualizer import Visualizer, VisualizerMsg, VMsgType



class Task:
	"""
		name：task名字
		volume：task需要几个node
		time：task需要花费的时间
	"""

	def __init__(self, name, volume, time):
		self.name = name
		self.volume = volume
		self.time = time

	def __str__(self):
		return "task_name: %s \t volume: %d \t time: %d " % (self.name, self.volume, self.time)


class RunningRecord:

	def __init__(self, name, volume, rest_time):
		self.name = name
		self.volume = volume
		self.rest_time = rest_time

	def __str__(self):
		return "task_name: %s \t volume: %d \t rest_time: %d " % (self.name, self.volume, self.rest_time)


# 每个task运行时的记录
class Record:
	def __init__(self, task_id, start_location, end_location, including_nodes_size, rest_time):
		self.task_id = task_id
		self.start_location = start_location
		self.end_location = end_location
		self.including_nodes_size = including_nodes_size
		self.rest_time = rest_time

	def __str__(self):
		return "task_name: %s\tstart_location: %s\t end_location: %s\t nodes: %d\t rest_time: %d" % (
			self.task_id, str(self.start_location), str(self.end_location), self.including_nodes_size, self.rest_time)


@unique
class FitMethodType(Enum):
	FIRST_FIT = 'first_fit'
	BEST_FIT = 'best_fit'
	WORST_FIT = 'worst_fit'
	NEXT_FIT = 'next_fit'


# 总体思路：一个生产者用来生产任务，一个消费者用来消费任务，即为任务找到合适的nodes，最后一个定时任务每过一秒就把所有正在执行的任务剩余时间减一
class TaskSchedulerFor3D:
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
	def __init__(self, size, arrival_rate=10, task_num=5, method_name=FitMethodType.FIRST_FIT,
	             enable_back_filling=False, max_cost_time=5, enable_visualize=False):
		self.max_cost_time = max_cost_time
		self.enable_back_filling = enable_back_filling
		self.task_num = task_num
		self.task_arrival_rate = arrival_rate
		self.taskQueue = Queue(maxsize=0)
		self.remaining_task_num = self.task_num

		self.visualQueue = Queue(maxsize=0)
		self.first_display = True
		self.enable_visualize = enable_visualize
		self.visualizer = None

		self.hpc = np.zeros(size, dtype=int)
		self.job_name_matrix = np.zeros(size, dtype=int)
		self.total_nodes = size[0] * size[1] * size[2]

		self.lock = threading.Condition()
		self.scheduler = BlockingScheduler()
		self.time_job_id = 'time processing'

		self.utilization_list = []
		self.utilization_file_name = ''
		self.method_name = method_name

		self.task_list = []
		self.running_task_list = []
		# 包括所有正在运行和已经运行完的task_id
		self.history_task_set = set()

		# 本次back_filling结束倒计时
		self.counter = 0
		self.last_location_for_back_fill = (-1, 0, 0)
		# self.test_set = set()

		self.test_flag = True

		self.data_init()

	# 完成所有数据的初始化工作
	def data_init(self):
		# self.job_name_matrix_init()
		pass

	# if self.enable_back_filling:
	# 	self.record_list_init()

	def test_generate_task(self):
		file_path = './task.txt'
		f = open(file_path)
		lines = f.readlines()
		f.close()
		for line in lines:
			values = line.split(',')
			task = Task(values[0], int(values[1]), int(values[2]))
			self.taskQueue.put(task)
			if self.enable_back_filling:
				# self.task_dict[values[0]] = task
				self.task_list.append(task)

	# 生产任务
	def generate_task(self):
		arrival_times = self.get_arrival_times()
		print(arrival_times)
		i = 1
		last_time = 0
		for t in arrival_times:
			time.sleep(t - last_time)
			last_time = t
			j_name = 'j' + str(i)
			volume = random.randint(1, self.total_nodes)
			cost_time = random.randint(1, self.max_cost_time)
			task = Task(j_name, volume, cost_time)
			self.taskQueue.put(task)
			if self.enable_back_filling:
				# self.task_dict[j_name] = task
				self.task_list.append(task)
			i = i + 1

	# 根据柏松过程生成每个任务的到达时间，后续任务会根据这个时间生成
	def get_arrival_times(self):
		arrive_times = []
		t = 0
		for i in range(self.task_num):
			t += random.expovariate(self.task_arrival_rate)
			arrive_times.append(t)
		return arrive_times

	# 测试方法，不用在意
	def printTask(self):
		size = self.taskQueue.qsize()
		for i in range(size):
			task = self.taskQueue.get()
			print(task)
			self.taskQueue.put(task)
		print(self.taskQueue.qsize())

	def create_columns_list(self):
		columns_list = []
		[_, _, columns] = self.hpc.shape
		for i in range(columns):
			columns_list.append(i)
		return columns_list

	def print_matrix(self):
		self.lock.acquire()
		try:
			[height, rows, columns] = self.hpc.shape
			columns_list = self.create_columns_list()
			for h in range(height-1, -1, -1):
				if columns_list[0] != 0:
					columns_list.reverse()
				for i in range(rows):
					for j in columns_list:
						print('%2d' % self.hpc[h, i, j], end=' ')
					print(end='\t')
					columns_list.reverse()
				print()

		finally:
			self.lock.release()

	def test_schedule(self):
		while self.remaining_task_num > 0:
			print("I am in")
			task = self.taskQueue.get()
			if task.name == 'j4':
				time.sleep(1)
			is_find = self.best_fit(task)
			# 如果找到合适位置了就把未安排任务数量减一
			if is_find:
				self.remaining_task_num = self.remaining_task_num - 1

	# 调度所有任务
	# 其实可以加个变量表示现在剩余的空nodes数量，如果当前任务需要的数量比空nodes数量多，直接开启back_filling即可，不需要再尝试一次fit了
	def schedule(self):
		self.utilization_file_name = 'test.csv'
		is_find = False
		last_location = (-1, 0, 0)
		locations = []
		while self.remaining_task_num > 0:
			# print('remaining task num: {}'.format(self.remaining_task_num))
			task = self.taskQueue.get()
			if task.name in self.history_task_set:
				continue
			else:
				for t in self.task_list:
					if t.name == task.name:
						self.task_list.remove(t)
			print(task)
			# if self.enable_back_filling and task.name not in self.task_dict:
			# 	continue
			while not is_find:
				if self.method_name == FitMethodType.FIRST_FIT:
					is_find, locations = self.first_fit(task)
				elif self.method_name == FitMethodType.BEST_FIT:
					is_find, locations = self.best_fit(task)
				elif self.method_name == FitMethodType.WORST_FIT:
					is_find, locations = self.worst_fit(task)
				elif self.method_name == FitMethodType.NEXT_FIT:
					is_find, locations = self.next_fit(task, last_location)
				if not is_find:
					self.do_if_not_find(task)
			self.do_after_find(task, locations)
			if self.method_name == FitMethodType.NEXT_FIT:
				last_location = locations[task.volume - 1]

			is_find = False
			locations.clear()
			self.test_flag = True

	# 每个已安排任务的剩余执行时间减一
	def time_process(self):
		# print('time process开始时间：{}'.format(time.time()))
		changed = False
		need_visualize_change = False
		not_empty_cell_num = 0
		self.lock.acquire()
		try:
			[height, rows, columns] = self.hpc.shape
			for h in range(height):
				for i in range(rows):
					for j in range(columns):
						if self.hpc[h, i, j] > 0:
							self.hpc[h, i, j] = self.hpc[h, i, j] - 1
							changed = True
							not_empty_cell_num = not_empty_cell_num + 1
							if self.enable_visualize and self.hpc[h, i, j] == 0:
								need_visualize_change = True
						if self.hpc[h, i, j] == 0:
							self.job_name_matrix[h, i, j] = 0

			# print("过了一秒，所有任务剩余执行时间减一：")
			# self.print_matrix()
			# print()

			if self.enable_visualize and need_visualize_change:
				self.post_process_voxels()

			if not_empty_cell_num > 0:
				self.utilization_list.append(not_empty_cell_num / self.total_nodes)

			if self.enable_back_filling:
				# self.process_record_time()
				# self.merge_empty_records()
				self.check_and_update_record()
				if self.counter > 0:
					self.counter = self.counter - 1

			if self.remaining_task_num == 0 and not changed:
				print("已完成所有任务")
				self.stop_visualize()
				self.scheduler.remove_job(self.time_job_id)
				self.scheduler.shutdown(wait=False)
				# self.write_utilization_to_file()
				return
		finally:
			self.lock.release()

	def stop_visualize(self):
		if self.enable_visualize:
			n_voxels = np.zeros(self.hpc.shape, dtype=bool)
			vMsg = VisualizerMsg(VMsgType.STOP, n_voxels)
			self.visualQueue.put(vMsg)

	def check_and_update_record(self):
		removing_id_list = []
		for i, r in enumerate(self.running_task_list):
			if r.rest_time - 1 <= 0:
				removing_id_list.append(i)
			else:
				r.rest_time = r.rest_time - 1
		if len(removing_id_list) > 0:
			new_running_task = [t for i, t in enumerate(self.running_task_list) if i not in removing_id_list]
			self.running_task_list = new_running_task

	def write_utilization_to_file(self):
		with open('./old_result/' + self.utilization_file_name, 'a') as f:
			for u in self.utilization_list:
				f.write(str(u))
				f.write('\n')
		self.utilization_list.clear()

	def do_after_find(self, task, locations):
		# 如果已经找到，那么将对应的Nodes的数值赋为任务所需的执行时间
		print('已安排{}个任务，当前正在安排{}, 需要{}个nodes, 该task需要花费时间为{}个单位时间:'
		      .format(self.task_num - self.remaining_task_num, task.name, task.volume, task.time))
		self.update_hpc(locations, task)
		# self.print_matrix()

	def do_if_not_find(self, task):
		if self.enable_back_filling:
			# self.test_set.add(task.name)
			self.start_back_filling(task)

	def get_start_and_end_location(self, locations):
		if locations is not None:
			return locations[0], locations[-1]
		return ()

	# 用first fit策略为一个任务分配Nodes
	def first_fit(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		count = 0  # 统计当前已经找到的空的Node个数
		locations = []  # 保存所有当前已经找到的空的Node的位置
		columns_list = self.create_columns_list()
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)
		self.lock.acquire()
		try:
			for i in range(rows):
				for j in columns_list:
					for h in range(height):
						if count >= task.volume:
							break
						# 如果一个Node还没有任务且目前任务还没有找到足够的Nodes，那么就记录当前Node
						if self.hpc[h, i, j] <= judge_state:
							if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
							                                                 end_location) and task.time > waiting_time:
								count = 0
								locations.clear()
								continue
							count = count + 1
							locations.append((h, i, j))
						# 如果当前node的任务剩余时间大于0，那么就要重新进行统计了，因为当前结点正在执行任务，没有连续的Nodes了
						elif self.hpc[h, i, j] > judge_state:
							count = 0
							locations.clear()
					if count >= task.volume:
						break
				if count >= task.volume:
					break
				columns_list.reverse()

			# 如果遍历完所有Nodes后还是没有找到合适的Nodes序列
			if count < task.volume:
				return False, []

			return True, locations
		finally:
			self.lock.release()

	# 用next fit策略为一个任务分配Nodes
	def next_fit(self, task, last_location, judge_state=0, is_back_filling_task=False, waiting_time=0,
	             preserve_locations=None):
		self.lock.acquire()
		try:
			[height, rows, columns] = self.hpc.shape
			count = 0  # 统计当前已经找到的空的Node个数
			locations = []  # 保存所有当前已经找到的空的Node的位置

			# 尝试找到搜索起点
			found, next_location = self.next_location(last_location)

			if not found:
				# 说明搜索起点超出搜索范围了，从头开始找
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
						# 如果一个Node还没有任务且目前任务还没有找到足够的Nodes，那么就记录当前Node
						if self.hpc[h, i, j] <= judge_state:
							if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
							                                                 end_location) and task.time > waiting_time:
								count = 0
								locations.clear()
								continue
							count = count + 1
							locations.append((h, i, j))
						# 如果当前node的任务剩余时间大于0，那么就要重新进行统计了，因为当前结点正在执行任务，没有连续的Nodes了
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
			# 如果遍历完所有Nodes后还是没有找到合适的Nodes序列
			if count < task.volume:
				result, locations = self.sub_process(task, judge_state, is_back_filling_task, waiting_time,
				                                     preserve_locations)
				if result:
					return result, locations
				else:
					return False, []
			else:
				return True, locations
		finally:
			self.lock.release()

	def sub_process(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		count = 0  # 统计当前已经找到的空的Node个数
		locations = []  # 保存所有当前已经找到的空的Node的位置
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
					# 如果一个Node还没有任务且目前任务还没有找到足够的Nodes，那么就记录当前Node
					if self.hpc[h, i, j] <= judge_state:
						if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
						                                                 end_location) and task.time > waiting_time:
							count = 0
							locations.clear()
							continue
						count = count + 1
						locations.append((h, i, j))
					# 如果当前node的任务剩余时间大于0，那么就要重新进行统计了，因为当前结点正在执行任务，没有连续的Nodes了
					elif self.hpc[h, i, j] > judge_state:
						count = 0
						locations.clear()
				if break_flag:
					break
			if break_flag:
				break
			columns_list.reverse()

		# 如果遍历完所有Nodes后还是没有找到合适的Nodes序列
		if count < task.volume:
			return False, ()
		else:
			return True, locations

	# best fit策略为一个任务分配Nodes
	def best_fit(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		old_count = sys.maxsize  # 上一次找到的合适的Node个数
		old_locations = []  # 上一次找到的所有Node位置
		cur_count = 0  # 统计当前已经找到的空的Node个数
		locations = []  # 保存所有当前已经找到的空的Node的位置
		flag = False
		columns_list = self.create_columns_list()
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)
		self.lock.acquire()
		try:
			for i in range(rows):
				for j in columns_list:
					for h in range(height):
						# 如果一个Node还没有任务，那么就记录当前Node
						if self.hpc[h, i, j] <= judge_state:
							if is_back_filling_task and self.judge_in_middle((h, i, j), start_location,
							                                                 end_location) and task.time > waiting_time:
								# 如果正好目前找到的nodes个数和task需要的nodes数相等，结束查找
								if cur_count == task.volume:
									old_count = cur_count
									old_locations = locations[:]
									flag = True
									break
								# 如果目前找到的nodes个数大于task的需求，那么和上一次的比较看是否留下这次结果
								elif task.volume < cur_count < old_count:
									old_count = cur_count
									cur_count = 0
									old_locations = locations[:]
									locations.clear()
								# 小于task的需求就直接舍弃这次结果
								else:
									cur_count = 0
									locations.clear()
								continue
							cur_count = cur_count + 1
							locations.append((h, i, j))
						# 如果当前node的任务剩余时间大于0，那么就要重新进行统计了，因为当前结点正在执行任务，没有连续的Nodes了
						elif self.hpc[h, i, j] > judge_state:
							# 如果正好目前找到的nodes个数和task需要的nodes数相等，结束查找
							if cur_count == task.volume:
								old_count = cur_count
								old_locations = locations[:]
								flag = True
								break
							# 如果目前找到的nodes个数大于task的需求，那么和上一次的比较看是否留下这次结果
							elif task.volume < cur_count < old_count:
								old_count = cur_count
								cur_count = 0
								old_locations = locations[:]
								locations.clear()
							# 小于task的需求就直接舍弃这次结果
							else:
								cur_count = 0
								locations.clear()
					if flag:
						break
				columns_list.reverse()
				if flag:
					break
			# 防止漏下后面全是空Nodes的情况，所以要再统计一次
			if task.volume <= cur_count < old_count:
				old_count = cur_count
				cur_count = 0
				old_locations = locations[:]
				locations.clear()

			# 判断是否找到合适的Nodes
			found = False
			if flag:
				found = True
			# 所有Nodes全为空闲的情况
			elif cur_count == self.total_nodes:
				found = True
				old_count = cur_count
				old_locations = locations[:]
			elif old_count != sys.maxsize:
				found = True
			# 如果遍历完所有Nodes后还是没有找到合适的Nodes序列，那么将当前任务放到队尾等待重新调度
			if not found:
				return found, []

			return found, old_locations[:task.volume]
		finally:
			self.lock.release()

	def worst_fit(self, task, judge_state=0, is_back_filling_task=False, waiting_time=0, preserve_locations=None):
		[height, rows, columns] = self.hpc.shape
		old_count = -1  # 上一次找到的合适的Node个数
		old_locations = []  # 上一次找到的所有Node位置
		cur_count = 0  # 统计当前已经找到的空的Node个数
		locations = []  # 保存所有当前已经找到的空的Node的位置
		flag = False
		columns_list = self.create_columns_list()
		start_location = None
		end_location = None
		if is_back_filling_task:
			[start_location, end_location] = self.get_start_and_end_location(preserve_locations)
		self.lock.acquire()
		try:
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
		finally:
			self.lock.release()

	def update_hpc(self, locations, task):
		self.lock.acquire()
		try:
			# 因为有可能找到的连续Nodes数量大于task的需求，所以要做判断，满足task需求了即可
			for i, t in enumerate(locations):
				if i < task.volume:
					self.hpc[t[0], t[1], t[2]] = task.time
					self.job_name_matrix[t[0], t[1], t[2]] = int(task.name[1:])
				else:
					break
			self.remaining_task_num = self.remaining_task_num - 1
			if self.enable_back_filling:
				# self.task_dict.pop(task.name)
				# # 向record list中插入新记录
				# start_location = locations[0]
				# end_location = locations[task.volume - 1]
				# record = Record(task.name, start_location, end_location, task.volume, task.time)
				# self.insert_record(record)
				record = RunningRecord(task.name, task.volume, task.time)
				self.running_task_list.append(record)
				self.running_task_list.sort(key=lambda x: x.rest_time, reverse=False)

				self.history_task_set.add(task.name)
			if self.enable_visualize:
				self.post_process_voxels()
			return locations[task.volume - 1]
		finally:
			self.lock.release()

	def post_process_voxels(self):
		voxels = np.zeros(self.hpc.shape, dtype=bool)
		[height, rows, columns] = self.hpc.shape
		for i in range(rows):
			for j in range(columns):
				for h in range(height):
					if self.hpc[h, i, j] > 0:
						voxels[j, i, h] = True
					else:
						voxels[j, i, h] = False
		vMsg = VisualizerMsg(VMsgType.CONTINUE, voxels)
		self.visualQueue.put(vMsg)

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

	def get_all_locations(self, start_location, end_location):
		[start_h, start_row, start_col] = start_location
		[end_h, end_row, end_col] = end_location
		[height, rows, columns] = self.hpc.shape

		result_list = []
		if start_row % 2 == 0:
			columns_list = [x for x in range(start_col, columns)]
		else:
			columns_list = [x for x in range(start_col, -1, -1)]
		height_list = [x for x in range(start_h, height)]
		break_flag = False

		for i in range(start_row, end_row + 1):
			for j in columns_list:
				for h in height_list:
					r, t = self.next_location((h, i, j))
					if r:
						[next_h, next_row, next_col] = t
						if next_h == end_h and next_row == end_row and next_col == end_col:
							break_flag = True
					result_list.append((h, i, j))
					if break_flag:
						break
				if break_flag:
					break
				if len(height_list) < height:
					for value in range(start_h):
						height_list.insert(value, value)
			if break_flag:
				break
			if len(columns_list) < columns:
				if start_row % 2 == 0:
					for value in range(start_col):
						columns_list.insert(value, value)
				else:
					columns_list.reverse()
					for value in range(start_col + 1, columns):
						columns_list.append(value)
					columns_list.reverse()
			columns_list.reverse()

		result_list.append(end_location)
		return result_list

	def universal_find_nodes_and_min_wait_time(self, first_task):
		locations = []
		min_wait_time = sys.maxsize
		for r in self.running_task_list:
			result, locations = self.first_fit(first_task, judge_state=r.rest_time)
			if result:
				# print('wait_time: {}\tlocations: {}'.format(r.rest_time, locations))
				# self.test_flag = False
				min_wait_time = r.rest_time
				break
		return locations, min_wait_time

	def start_back_filling(self, first_task):
		# if self.test_flag:
		# wait_time, locations = self.find_nodes_and_min_wait_time(first_task)
		# print('wait_time: {}\tlocations: {}'.format(wait_time, locations))
		# self.test_flag = False
		self.lock.acquire()
		try:
			locations, self.counter = self.universal_find_nodes_and_min_wait_time(first_task)
			# while self.counter > 0:
			# 	self.back_fill_task(next_task=first_task, locations=locations)
			self.back_fill_task(locations, first_task)
			self.scheduler.add_job(self.back_fill_task, 'interval', seconds=1, id='back_fill_task',
			                       args=[locations, first_task])
			self.lock.wait()
		finally:
			self.lock.release()

	def back_fill_task(self, locations, first_task):
		self.lock.acquire()
		try:
			removing_task_id = []
			if self.counter == 0:
				self.scheduler.remove_job('back_fill_task')
				self.lock.notify()
				self.last_location_for_back_fill = (-1, 0, 0)
				return
			for i, t in enumerate(self.task_list):
				if t.name in self.history_task_set:
					continue
				# found, res_locations = self.next_fit(task=t, is_back_filling_task=True, waiting_time=self.counter,
				#                                      preserve_locations=locations,
				#                                      last_location=self.last_location_for_back_fill)
				found, res_locations = self.first_fit(task=t, is_back_filling_task=True, waiting_time=self.counter,
				                                      preserve_locations=locations)
				# if first_task.name == 'j2':
				# 	print('cur_task:{}\tfound:{}'.format(t, found))
				if found:
					removing_task_id.append(i)
					self.do_after_find(t, res_locations)
					self.last_location_for_back_fill = res_locations[-1]

			removing_task_id.clear()
		# self.counter = self.counter - 1
		# time.sleep(1.00000001)
		finally:
			self.lock.release()

	def start_time_processing(self):
		self.scheduler.add_job(self.time_process, 'interval', seconds=1, id=self.time_job_id, max_instances=5)
		self.scheduler.start()

	# 仿真程序主入口
	def simulate(self):
		producer = threading.Thread(target=self.generate_task, name='producer')
		consumer = threading.Thread(target=self.schedule, name='consumer')
		producer.start()
		consumer.start()
		threading.Thread(target=self.start_time_processing, name='kk').start()
		if self.enable_visualize:
			if self.visualizer is None:
				self.visualizer = Visualizer(self.visualQueue)
			self.visualizer.start()
			# print("可视化开始")



if __name__ == '__main__':
	v = 5
	hpc_size = [v, v, v]
	task_arrival_rate = 5
	task_num = 10
	method_name = FitMethodType.FIRST_FIT
	max_cost_time = 10
	scheduler3D = TaskSchedulerFor3D(hpc_size, task_arrival_rate, task_num, method_name,
	                                 max_cost_time=max_cost_time, enable_back_filling=True, enable_visualize=True)
	scheduler3D.simulate()
