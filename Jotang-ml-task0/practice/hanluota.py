def move(n, a, b, c):
    if n == 1:
        print(a, '-->', c)
    else:
        # 将n-1个盘子从a借助c移动到b
        move(n - 1, a, c, b)
        # 将最底下的第n个盘子从a直接移动到c
        print(a, '-->', c)
        # 将n-1个盘子从b借助a移动到c
        move(n - 1, b, a, c)
move(3, 'A', 'B', 'C')