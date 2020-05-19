from enum import unique, Enum


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

	def __cmp__(self, other):
		if int(self.time) < int(other.time):
			return -1
		elif int(self.time) > int(other.time):
			return 1
		else:
			return 0

	def __lt__(self, other):
		return self.time < other.time

	def __le__(self, other):
		return self.time <= other.time

	def __str__(self):
		return "task_name: %s \t volume: %d \t time: %d " % (self.name, self.volume, self.time)


class RunningRecord:

	def __init__(self, name, volume, rest_time, wasted_nodes_locations=None):
		if wasted_nodes_locations is None:
			wasted_nodes_locations = []
		self.name = name
		self.volume = volume
		self.rest_time = rest_time
		self.wasted_nodes_locations = wasted_nodes_locations

	def __str__(self):
		return "task_name: %s \t volume: %d \t rest_time: %d" % (self.name, self.volume, self.rest_time)


@unique
class FitMethodType(Enum):
	FIRST_FIT = 'first_fit'
	BEST_FIT = 'best_fit'
	WORST_FIT = 'worst_fit'
	NEXT_FIT = 'next_fit'
