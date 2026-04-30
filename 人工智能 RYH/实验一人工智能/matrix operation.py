def MatrixAdd(A, B):
    a = len(A[0])
    E = [[0]*a for b in range(a)]
    i = 0
    j = 0
    while i < a:
        while j < a:
              E[i][j] = A[i][j] + B[i][j]
              j = j + 1
        i = i + 1
        j = 0
    print(E)
    return E


def MatrixMul(A, B):
    c = len(A[0])
    F = [[0]*c for b in range(c)]
    i = 0
    j = 0
    while i < c:
        while j < c:
            d = 0
            while d < c:
                F[i][j] = F[i][j] + A[i][d] * B[d][j]
                d = d + 1
            j = j + 1
        i = i + 1
        j = 0
    print(F)
    return F


C = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
D = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
MatrixAdd(C, D)
MatrixMul(C, D)
