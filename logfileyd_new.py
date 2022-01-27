import datetime
import calendar
import os
import time
import fireb

logfile = "logsv4.txt"
local_log_file = "locallogv1.txt"

if not os.path.isfile(logfile):
    write_file = open(logfile, "w")
    write_file.write("")
    write_file.close()

if not os.path.isfile(local_log_file):
    write_file = open(local_log_file, "w")
    write_file.write("")
    write_file.close()


def unix_converter(unix):
    return datetime.datetime.fromtimestamp(int(unix)).strftime('%Y-%m-%d %H:%M:%S')


def write_to_log(student_name, module, class_name, start_time, end_time, status):
    print("Logging %s" % student_name)
    current_unix_timestamp = calendar.timegm(time.gmtime())
    d = str(unix_converter(current_unix_timestamp))
    date = d[0:10]
    ct = str(time.strftime("%H%M%S", time.localtime()))
    CT = str(time.strftime("%H:%M", time.localtime()))

    text = ("Unix: " + str(current_unix_timestamp) +
            " StudentID: " + str(student_name) +
            " Classofthestudent: " + str(class_name) +
            " ModuleCode: " + str(module) +
            " date: " + str(date) +
            " Time: " + str(ct) +
            "\n")

    with open(logfile, "a") as write_log:
        write_log.write(text)

    with open(local_log_file, "a") as write_local_log:
        write_local_log.write(text)

    start_time = int(start_time)
    end_time = int(end_time)

    if start_time < 1000:
        start_time = str(start_time)
        start_time = "0" + start_time

    if end_time < 1000:
        end_time = str(end_time)
        end_time = "0" + end_time
    return str(CT), str(student_name), str(module), str(class_name), str(start_time), str(end_time), status


def logging(student_name, module, class_name, start_time, end_time, status):
    exist = False
    with open(logfile, "r") as read_log:
        lines = read_log.readlines()

    current_unix_timestamp = calendar.timegm(time.gmtime())
    date = str(unix_converter(current_unix_timestamp))
    date = date[0:10]

    record = str(lines)
    record = record.split(",")

    if date not in str(record[0]):  # clear if data is different
        print("Deleting old records")
        with open(logfile, "w") as log:
            log.write("")

    if lines:
        if module in str(record[0]) and class_name in str(record[0]):
            for n in range(len(lines)):
                b = record[n]
                if student_name in str(b):
                    return None
        else:
            with open(logfile, "w") as log:
                log.write("")
            temp = write_to_log(student_name, module, class_name, start_time, end_time, status)
            return temp

        if not exist:
            temp = write_to_log(student_name, module, class_name, start_time, end_time, status)
            return temp
    else:
        temp = write_to_log(student_name, module, class_name, start_time, end_time, status)
        return temp

