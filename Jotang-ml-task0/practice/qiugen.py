# 计算一元二次方程的根
def quadratic(a,b,c):
    delta = b**2 - 4*a*c
    if delta < 0:
        return "无实数解"
    elif delta == 0:
        return -b/(2*a)
    else:
        x1 = (-b + delta**0.5)/(2*a)
        x2 = (-b - delta**0.5)/(2*a)
        return x1,x2
print(quadratic(4,6,2))