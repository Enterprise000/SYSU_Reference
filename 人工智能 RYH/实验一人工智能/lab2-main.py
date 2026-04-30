class StuData:
    len = 0
    data = [['z']*4 for a in range(10)]

    def __init__(self):
        f = open("student_data.txt", mode='r')
        line = f.readline()
        i = 0
        j = 0
        while i < 10:
            StuData.data[i][3] = 100
            i = i + 1
        i = 0
        while line:
            list1 = line.split()
            StuData.len = StuData.len + 1
            while j < 4:
                if j < 3:
                  StuData.data[i][j] = list1[j]
                if j == 3:
                  StuData.data[i][j] = int(list1[j])
                #print("current list: ", i, j,StuData.data[i][j])
                j = j + 1
            i = i + 1
            j = 0
            line = f.readline()
        print("data:", StuData.data)

    def AddData(self, name, stu_num, gender, age):
        StuData.data[StuData.len][0] = name
        StuData.data[StuData.len][1] = stu_num
        StuData.data[StuData.len][2] = gender
        StuData.data[StuData.len][3] = int(age)
        print("add: ", StuData.data)

    def SortData(self, standard):
        if standard == 'name':
            StuData.data = sorted(StuData.data, key=lambda x: x[0], reverse=False)
        if standard == 'stu_num':
            StuData.data = sorted(StuData.data, key=lambda x: x[1], reverse=False)
        if standard == 'gender':
            StuData.data = sorted(StuData.data, key=lambda x: x[2], reverse=False)
        if standard == 'age':
            StuData.data = sorted(StuData.data, key=lambda x: x[3], reverse=False)
        print("sorted data: ", StuData.data)

    def ExportFile(self):
        a = 0
        with open('new_data.txt', 'w') as file:
          while a <= StuData.len:
            b = StuData.data[a][0] + " " + StuData.data[a][1] + " " + StuData.data[a][2] + " " + str(StuData.data[a][3
            ]) + '\n'
            file.write(b)
            #print(b)
            a = a + 1


student = StuData()
student.SortData('stu_num')
student.AddData('Lisa', '156', 'F', '24')
student.ExportFile()
