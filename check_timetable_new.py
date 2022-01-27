import xlrd
import time

file = "roomTimetable.xls"


def get_student_list():
    wb = xlrd.open_workbook(file)
    lesson_sheet = wb.sheet_by_index(0)
    name_sheet = wb.sheet_by_index(1)

    t = time.time()
    current_day = time.ctime(t).split(" ")[0]
    current_time = time.ctime(t).split(" ")[-2].replace(":", "")[:-2]
    early_time = time.ctime(t + 600).split(" ")[-2].replace(":", "")[:-2]
    student_list = []

    if int(current_time) <= 2400:

        current_row = 0
        while current_day not in lesson_sheet.cell_value(current_row, 0):
            current_row += 1

        current_col = 1
        while 1:
            _, end = lesson_sheet.cell_value(0, current_col).split("-")
            if current_time < end:
                break
            current_col += 1

        current_cell = lesson_sheet.cell_value(current_row, current_col)

        if current_cell != "":
            starting_col = current_col
            ending_col = current_col

            while current_cell == lesson_sheet.cell_value(current_row, starting_col):
                starting_col -= 1
            while current_cell == lesson_sheet.cell_value(current_row, ending_col):
                ending_col += 1

            starting_col += 1
            ending_col -= 1

            start_time = lesson_sheet.cell_value(0, starting_col)[:4]
            end_time = lesson_sheet.cell_value(0, ending_col)[5:]
            module, _, class_name = current_cell.split("\n")
            late_time = int(start_time) + 15

            z = 0
            while not class_name == name_sheet.cell_value(0, z):
                z += 1

            for i in range(name_sheet.nrows):
                if i != 0:
                    if name_sheet.cell_value(i, z) != "":
                        student_list.append(name_sheet.cell_value(i, z))

            if int(current_time) >= late_time:
                return student_list, class_name, module, 1, start_time, end_time

            if early_time >= start_time and int(current_time) < late_time:
                return student_list, class_name, module, 0, start_time, end_time
    else:
        return None, None, None, 2, None, None


def getRoomName():
    wb = xlrd.open_workbook(file)
    lesson_sheet = wb.sheet_by_index(0)
    return lesson_sheet.cell_value(0, 0)


def getLateTime():
    wb = xlrd.open_workbook(file)
    lesson_sheet = wb.sheet_by_index(0)

    t = time.time()
    current_day = time.ctime(t).split(" ")[0]
    current_time = time.ctime(t).split(" ")[-2].replace(":", "")[:-2]

    current_row = 0
    while current_day not in lesson_sheet.cell_value(current_row, 0):
        current_row += 1

    current_col = 1
    while 1:
        _, end = lesson_sheet.cell_value(0, current_col).split("-")
        if current_time < end:
            break
        current_col += 1

    current_cell = lesson_sheet.cell_value(current_row, current_col)

    if int(current_time) <= 2400:
        starting_col = current_col
        while current_cell == lesson_sheet.cell_value(current_row, starting_col):
            starting_col -= 1
        starting_col += 1

        start_time = lesson_sheet.cell_value(0, starting_col)[:4]
        late_time = int(start_time) + 15
        return late_time


def get_HumanName():
    wb = xlrd.open_workbook(file)
    name_sheet = wb.sheet_by_index(1)

    HumanNames = []
    for i in range(name_sheet.ncols):
        for j in range(name_sheet.nrows):
            if j != 0:
                name = name_sheet.cell_value(j, i)
                if name is not "" and name not in HumanNames:
                    HumanNames.append(name)
    HumanNames.sort()

    return HumanNames
