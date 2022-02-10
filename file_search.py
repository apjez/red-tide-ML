import datetime as dt
import math

#binary search to find correct file in list based on search date/time
#list is assumed to already be in order
def file_search(searchdatetime, file_list):
	found = False
	return_file = ''
	list_start = 0
	list_end = len(file_list)
	timediff = dt.timedelta(hours=3)
	count = 0
	last_list_start = -1
	last_list_end = -1
	while(found == False):
		mid_point = math.floor((list_end+list_start)/2)

		filedatetime_string = file_list[mid_point][0:19]
		year = int(filedatetime_string[0:4])
		month = int(filedatetime_string[5:7])
		day = int(filedatetime_string[8:10])
		hour = int(filedatetime_string[11:13])
		minute = int(filedatetime_string[14:16])
		second = int(filedatetime_string[17:19])

		mid_point_datetime = dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

		if(searchdatetime > mid_point_datetime):
			mid_point_diff = searchdatetime-mid_point_datetime
		else:
			mid_point_diff = mid_point_datetime-searchdatetime

		#point has been found or list has been exhausted
		if(mid_point_diff < timediff or (last_list_start==list_start and last_list_end==list_end)):
			found = True
			return_file = file_list[mid_point]
		else:
			last_list_start = list_start
			last_list_end = list_end
			if(searchdatetime < mid_point_datetime):
				list_end = mid_point
			else:
				list_start = mid_point

		count = count + 1
	return return_file