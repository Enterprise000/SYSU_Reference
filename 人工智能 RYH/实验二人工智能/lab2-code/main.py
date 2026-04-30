import re
# 将每个谓词转化为（名称，参数）的格式,输入一个列表，返回一个二维列表


def trans(string):
    tmp_list = re.split(r'[(,]', string)
    # print(tmp_list)
    tmp_list[len(tmp_list)-1] = tmp_list[len(tmp_list)-1].replace(")", "")
    # print(tmp_list)
    return tmp_list


def anti_trans(list7):
      str1 = list7[0] + '('
      g = 1
      while g < len(list7)-1:
          str1 = str1 + list7[g] + ','
          g = g + 1
      str1 = str1 + list7[len(list7)-1] + ')'
      return str1


def equals(l1,l2):
    #print(l1,l2)
    if len(l1) != len(l2):
        return 0
    else:
        i = 1
        while i < len(l1):
            if l1[i] != l2[i]:
                return 0
            i = i+1
        return 1

def v_num(list8):
    num = 0
    v_list = ['xx', 'yy', 'x', 'y', 'z', 'w', 'u', 'v']
    for i in range(len(list8)):
        if list8[i] in v_list:
            num = num + 1
    return num


def differ(list1):
    my_set = set(list1)
    list1 = list(my_set)
    return list1

def unifier(l1, l2):
    v_list = ['xx', 'yy', 'x', 'y', 'z', 'w', 'u', 'v']
    s = '0'
    for t in range(len(l1)):
        #不能合一，返回0
        if v_num(l1) > 1 or v_num(l2) > 1:
            return s
        if t > 0 and l1[t] not in v_list and l2[t] not in v_list and l1[t] != l2[t] :
            return s
        #需要合一，且只有一个变量
        if l1[t] != l2[t] and (l1[t] in v_list or l2[t] in v_list):
            if l1[t] in v_list:
                if l2[t] in v_list:
                    l1[t] = l2[t]
                    s = l2[t]
                    return s
                else:
                    l1[t] = l2[t]
                    s = l2[t]
                    return s
            else:
                if l2[t] in v_list:
                    l2[t] = l1[t]
                    s = l1[t]
                    return s
        #除了符号都相同，返回1
        if (l1[0] == '~' + l2[0] or l2[0] == '~' + l1[0]) and equals(l1,l2) == 1:
            s = '1'
    return s


def unify(list5, target):
    v_list = ['xx', 'yy', 'x', 'y', 'z', 'w', 'u', 'v']
    list6 = list5.copy()
    for d in range(len(list5)):
        #转化为方便处理的形式
        if list5[d] != ' ':
          temp = trans(list5[d])
          for f in range(len(temp)):
            if temp[f] in v_list:
                #将变量变成目标量（变量或常量）
                temp[f] = target
          list6[d] = anti_trans(temp)
    return list6



# 判断两个谓词的名称是否一致
def same(string1, string2):
    q = string1.find("(")
    w = string2.find("(")
    if string1[0] == '~' and string2[0] != '~':
        string1 = string1[1:q]
        string2 = string2[0:w]
    if string2[0] == '~' and string1[0] != '~':
        string1 = string1[0:q]
        string2 = string2[1:w]
    if string1 == string2:
        return 1
    else:
        return 0


def all_space(list3):
    r = 1
    for item in list3:
        if item != ' ':
            r = 0
    return r


def ResolutionFOL(clause_list1):
# 循环遍历每个tuple的元素
    i = 0
    x = len(clause_list1)
    while i < x:
        j = 0
        cur_len = len(clause_list1)
        while j < len(clause_list1[i]):
            if len(clause_list1[i]) == 0:
                break
            k = 0
            while k < len(clause_list1):
                l = 0
                while l < len(clause_list1[k]):
                        # 选出要比较的
                        tmp_c1 = clause_list1[i][j]
                        tmp_c2 = clause_list1[k][l]
                        if all_space(tmp_c1) == 0 and all_space(tmp_c2) == 0:
                          a = same(tmp_c1, tmp_c2)
                          if a == 1 and tmp_c1[0] != tmp_c2[0]:
                              list1 = trans(tmp_c1)
                              list2 = trans(tmp_c2)
                              v_list = ['xx', 'yy', 'x', 'y', 'z', 'w', 'u', 'v']
                              v1 = '0'
                              v2 = '0'
                              for d in range(len(list1)):
                                  if list1[d] in v_list:
                                      v1 = list1[d]
                              for d in range(len(list2)):
                                  if list2[d] in v_list:
                                      v2 = list2[d]
                              # unify,归结，输出
                              y = unifier(list1, list2)
                              if y == '1':
                                  #除了符号都相同，直接归结
                                  tlist1 = clause_list1[i].copy()
                                  tlist2 = clause_list1[k].copy()
                                  tlist1[j] = ' '
                                  tlist2[l] = ' '
                                  tlist1.extend(tlist2)
                                  tlist1 = differ(tlist1)
                                  clause_list1.append(tlist1)
                                  show(clause_list1, i, j, k, l, tlist1,y,v1,v2)
                                  if all_space(tlist1) == 1:
                                      print("结束")
                                      return
                              if y != '1' and y != '0' and list1[0] != list2[0]:
                                #赋值
                                 tlist1 = unify(clause_list1[i], y)
                                 tlist2 = unify(clause_list1[k], y)
                                 tlist1[j] = ' '
                                 tlist2[l] = ' '
                                 tlist1.extend(tlist2)
                                 tlist1 = differ(tlist1)
                                 clause_list1.append(tlist1)
                                 show(clause_list1,i,j,k,l,tlist1,y,v1,v2)
                                 if all_space(tlist1) == 1:
                                      print("结束")
                                      return
                        l = l + 1
                k = k + 1
            j = j + 1
        i = i + 1
        if cur_len == len(clause_list1):
            print("不能归结")
            return
    return

def letter(list, a, b):
    if len(list) == 1:
        str1 = str(a)
        return str1
    else:
        if b == 0:
            str1 = str(a) + 'a'
            return str1
        if b == 1:
            str1 = str(a) + 'b'
            return str1
        if b == 2:
            str1 = str(a) + 'c'
            return str1
        else:
            str1 = str(a) + 'd'
    return str1


def show(clause_list1,i,j,k,l,tlist3,y,v1,v2):
    v_list = ['xx', 'yy', 'x', 'y', 'z', 'w', 'u', 'v']
    tlist3.remove(' ')
    str1 = letter(clause_list1[i], i, j)
    str2 = letter(clause_list1[k], k, l)
    if v1 !='0':
        str3 = "R[" + str1 + ',' + str2 + "]=" + '{' + v1 + '=' + y + '}' + str(tuple(tlist3))
        print(str3)
    else:
        if v2 != '0':
            str3 = "R[" + str1 + ',' + str2 + "]=" + '{' + v1 + '=' + y + '}' + str(tuple(tlist3))
            print(str3)
        else:
            str3 = "R[" + str1 + ',' + str2 + "]=" + str(tuple(tlist3))
            print(str3)


# 子句数目和每个字句里的谓词数目
print("请输入子句集长度和每个子句中的原子数，每次输入需要换行")
num_clause = int(input())
num_words = []
for i in range(num_clause):
    num_words.append(int(input()))

# 输入KB,分割KB，存为元组和集合,另外存一个列表方便操作
print("请输入字句集，注意不要输入'KB='和大括号,并且所有的输入需要在一行内完成")
s = input()
origin = (re.findall(r'~?\w+\(\w+\,*\w*\)', s))
clause_set = set()
clause_list = list()
b = 0
a = 0
while b < sum(num_words):
    temp_list = origin[b:b+num_words[a]]
    temp_tuple = tuple(temp_list)
    print(temp_tuple)
    clause_set.add(temp_tuple)
    clause_list.append(temp_list)
    b = b + num_words[a]
    a = a+1
ResolutionFOL(clause_list)
