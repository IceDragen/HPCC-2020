import os

import numpy as np

def format_file(n):
	directory_path = './data/'
	pre_job_time_path = directory_path + 'pre_input_job_time_{}.txt'.format(n)
	pre_job_size_path = directory_path + 'pre_input_job_size_{}.txt'.format(n)
	input_job_size_path = directory_path + 'input_job_size_{}.txt'.format(n)
	input_job_time_path = directory_path + 'input_job_time_{}.txt'.format(n)
	pre_job_times = []
	pre_job_sizes = []
	input_job_sizes = []
	input_job_times = []

	with open(pre_job_time_path, 'r') as f:
		data = f.readlines()
		for line in data:
			pre_job_times = parse_data(line)
	with open(pre_job_size_path, 'r') as f:
		data = f.readlines()
		for line in data:
			pre_job_sizes = parse_data(line)
	with open(input_job_size_path, 'r') as f:
		data = f.readlines()
		for line in data:
			input_job_sizes = parse_data(line)
	with open(input_job_time_path, 'r') as f:
		data = f.readlines()
		for line in data:
			input_job_times = parse_data(line)
	os.remove(pre_job_time_path)
	os.remove(pre_job_size_path)
	os.remove(input_job_size_path)
	os.remove(input_job_time_path)
	pre_job_data = list(zip(pre_job_sizes, pre_job_times))
	# print('pre_job: {}'.format(len(pre_job_data)))
	input_job_data = list(zip(input_job_sizes, input_job_times))
	# print('input_job: {}'.format(len(input_job_data)))
	pre_job_data_path = directory_path + 'pre_job_data_{}.csv'.format(n)
	input_job_data_path = directory_path + 'input_job_data_{}.csv'.format(n)
	with open(pre_job_data_path, 'a+') as f:
		for t in pre_job_data:
			line = str(t[0]) + ',' + str(t[1])
			f.write(line + '\n')
	with open(input_job_data_path, 'a+') as f:
		for t in input_job_data:
			line = str(t[0]) + ',' + str(t[1])
			f.write(line + '\n')


def parse_data(line):
	result = []
	int_str = ''
	for c in line:
		if c.isdigit():
			int_str = int_str + c
		elif int_str != '':
			result.append(int(int_str))
			int_str = ''
	print(len(result))
	return result


for i in range(7, 12):
	format_file(i)