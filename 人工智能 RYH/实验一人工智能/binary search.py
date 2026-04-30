
def BinarySearch(nums, target):
    a = len(nums) // 2
    if target in nums:
        if target < nums[a]:
            nums = nums[0:a - 1]
            BinarySearch(nums, target)
        elif target == nums[a]:
            print("the index is: ", a)
            return a
        else:
            nums = nums[a + 1:]
            BinarySearch(nums, target)
    else:
        print("the number doesn't exist")
        return -1


alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
number = 3
BinarySearch(alist, number)
