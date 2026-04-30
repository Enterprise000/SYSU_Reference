def ReverseKeyValue(dict1):
    dict2 = dict([val, key]for key, val in dict1.items())
    print(dict2)
    return dict2

dict1 = {'Alice': '001', 'Bob': '002'}
ReverseKeyValue(dict1)