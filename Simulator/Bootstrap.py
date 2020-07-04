# -*- coding: utf-8 -*-
import functools
import os
import time
from multiprocessing.context import Process
from multiprocessing import Pool

from Simulator.Entities import Task, FitMethodType
from Simulator.Wireless import Wireless
from Simulator.Conventional import Conventional


class Bootstrap:
	# pre_job_prototypes: job list of initial workload
	# input_job_prototypes: one of N synthetic workloads
	# hpc_size: the size of HPC
	def __init__(self, v=24):
		self.pre_job_prototypes = []
		self.input_job_prototypes = []
		self.hpc_size = [v, v, v]

	# read job sizes and walltimes from data file
	def data_init(self, itr=1):
		data_path_prefix = './data/'
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

	# put all jobs into a queue
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

	# FCFS strategy
	def do_online_simulate_with_FCFS(self, n, rate, enable_bf, data_path, time_data_path, flag=True, itr=1):
		task_queue = self.get_task_queue(numbered=n, itr=itr)
		if flag:
			scheduler = Wireless(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                     time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                     arrival_rate=rate, enable_back_filling=enable_bf, st=2)
			scheduler.online_simulate_with_FCFS()
		else:
			scheduler = Conventional(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                         time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                         arrival_rate=rate, enable_back_filling=enable_bf, st=2)
			scheduler.online_simulate_with_FCFS()

	# SJF strategy
	def do_online_simulate_with_SJF(self, n, rate, enable_bf, data_path, time_data_path, flag=True, itr=1):
		task_queue = self.get_task_queue(numbered=n, itr=itr)
		if flag:
			scheduler = Wireless(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                     time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                     arrival_rate=rate, enable_back_filling=enable_bf, st=2,
			                     enable_time_sort=True)
			scheduler.online_simulate_with_SJF()
		else:
			scheduler = Conventional(size=self.hpc_size, task_queue=task_queue, data_path=data_path,
			                         time_path=time_data_path, method_name=FitMethodType.FIRST_FIT,
			                         arrival_rate=rate, enable_back_filling=enable_bf, st=2,
			                         enable_time_sort=True)
			scheduler.online_simulate_with_SJF()


if __name__ == '__main__':
	bootstrap = Bootstrap()
	data_path = './data.txt'
	time_data_path = './time.txt'
	bootstrap.do_online_simulate_with_FCFS(10, 7, True, data_path, time_data_path)
	# bootstrap.do_online_simulate_with_SJF(20, 5, True, data_path, time_data_path, flag=False)
