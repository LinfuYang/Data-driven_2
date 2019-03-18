class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        m = len(array)
        n = len(array[0])
        flag = False
        i = 1
        j = 1
        while i < m and j < n:
            if flag ==True:
                break
            elif array[i-1][j - 1] <= target and array[i][j] >= target:
                for k in range(j-1, j-1+6):
                    if k < n:
                        if array[i-1][k] == target:
                            flag = True

                    else:
                        if array[i][k-n] == target:
                            flag=True

            else:
                i += 1
                j += 1
        return flag
