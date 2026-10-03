from collections import Counter
c = Counter('shuobuxihuandoushijiade')
for ch in 'shuobuxihuandoushijiade':
    c[ch] = c[ch] + 1
print(c)
