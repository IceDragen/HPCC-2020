# -*- coding: utf-8 -*-
import functools
import os
import time
from multiprocessing.context import Process
from multiprocessing import Pool

from newsolution.Entity import Task, FitMethodType
from newsolution.Wireless import Wireless
from newsolution.Convention import Convention


class TaskDeco:

	def __init__(self, n, rate, itr, process) -> None:
		self.n = n
		self.rate = rate
		self.itr = itr
		self.p = process

	def __str__(self):
		return "n: %d \t rate: %d \t itr: %d " % (self.n, self.rate, self.itr)


class Bootstrap:
	def __init__(self, v=24):
		self.pre_job_prototypes = []
		self.input_job_prototypes = []
		self.hpc_size = [v, v, v]

	def data_init(self, itr=1):
		data_path_prefix = './newsolution/data/'
		pre_job_data_path = data_path_prefix + '/pre_job_data_{}.csv'.format(itr)
		input_job_data_path = data_path_prefix + '/input_job_data_{}.csv'.format(itr)
		self.pre_job_prototypes = self.get_task_prototype(pre_job_data_path)
		self.input_job_prototypes = self.get_task_prototype(input_job_data_path)

	def get_task_prototype(self, path):
		result = []
		with open(path, 'r') as f:
			data = f.readlines()
		for line in data:
			values = line.split(',')
			j_name = 'j0'
			volume = int(values[0])
			cost_time = int(values[1])
			task = Task(j_name, volume, cost_time)
			result.append(task)
		return result

	def get_task_queue(self, numbered=3, itr=1):
		self.data_init(itr)
		i = 1
		all_jobs = []
		for task in self.pre_job_prototypes:
			t = Task('j' + str(i), task.volume, task.time)
			i = i + 1
			all_jobs.append(t)
		for a in range(0, numbered):
			for task in self.input_job_prototypes:
				t = Task('j' + str(i), task.volume, task.time)
				i = i + 1
				all_jobs.append(t)
		return all_jobs

	def get_sorted_time_task_queue(self, numbered=3, itr=1):
		self.data_init(itr)
		i = 1
		all_jobs = []
		for task in self.pre_job_prototypes:
			t = Task('j' + str(i), task.volume, task.time)
			i = i + 1
			all_jobs.append(t)
		tmp_jobs = []
		for a in range(0, numbered):
			for task in self.input_job_prototypes:
				t = Task('j' + str(i), task.volume, task.time)
				i = i + 1
				tmp_jobs.append(t)
		tmp_jobs = sorted(tmp_jobs, key=functools.cmp_to_key(self.cmp))
		all_jobs.extend(tmp_jobs)
		return all_jobs

	def cmp(self, t1, t2):
		m = t1.time - t2.time
		# print(m)
		if m < 0:
			return -1
		elif m > 0:
			return 1
		else:
			return 0

	# online FCFS strategy
	def do_online_simulate(self, n, rate, enable_bf, data_path, time_data_path, flag=True, itr=1):
		task_queue = self.get_task_queue(numbered=n, itr=itr)
		if flag:
			scheduler = Wireless(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                     time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                     arrival_rate=rate, enable_back_filling=enable_bf, st=2)
			scheduler.online_simulate_with_FCFS()
		else:
			scheduler = Convention(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                       time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                       arrival_rate=rate, enable_back_filling=enable_bf, st=2)
			scheduler.online_simulate_with_FCFS()

	# online SJF strategy
	def do_online_simulate_with_sorted_time(self, n, rate, enable_bf, data_path, time_data_path, flag=True, itr=1):
		task_queue = self.get_task_queue(numbered=n, itr=itr)
		if flag:
			scheduler = Wireless(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                     time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                     arrival_rate=rate, enable_back_filling=enable_bf, st=2,
			                     enable_time_sort=True)
			scheduler.online_simulate_with_SJF()
		else:
			scheduler = Convention(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                       time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                       arrival_rate=rate, enable_back_filling=enable_bf, st=2,
			                       enable_time_sort=True)
			scheduler.online_simulate_with_SJF()



if __name__ == '__main__':
	bootstrap = Bootstrap()
	data_path = ''
	time_data_path = ''
	bootstrap.do_online_simulate(5, 5, True, data_path, time_data_path)
