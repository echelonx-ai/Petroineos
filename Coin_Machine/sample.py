sols = []
def possible_change(n,p=[],coins = [1,5,10]):
    if n == 0:
        global sols
        sols.append(p)
    else:
        for c in coins:
            if n - c >= 0:
                possible_change(n-c,p+[c],coins=coins)    

possible_change(0.5,coins=[0.5, 0.20, 0.10, 0.05])
print(sols)