# Function to find the total number of distinct ways to get a change of `target`
# from an unlimited supply of coins in set `S`
def count(S, n, target, lookup):
 
    # if the total is 0, return 1 (solution found)
    if target == 0:
        return 1
 
    # return 0 (solution does not exist) if total becomes negative,
    # no elements are left
    if target < 0 or n < 0:
        return 0
 
    # construct a unique key from dynamic elements of the input
    key = (n, target)
 
    # if the subproblem is seen for the first time, solve it and
    # store its result in a dictionary
    if key not in lookup:
 
        # Case 1. Include current coin `S[n]` in solution and recur
        # with remaining change `target-S[n]` with the same number of coins
        include = count(S, n, target - S[n], lookup)
 
        # Case 2. Exclude current coin `S[n]` from solution and recur
        # for remaining coins `n-1`
        exclude = count(S, n - 1, target, lookup)
 
        # assign total ways by including or excluding current coin
        lookup[key] = include + exclude
 
    # return solution to the current subproblem
    return lookup[key]
 
 
# Coin Change Problem
if __name__ == '__main__':
 
    S = [0.5, 0.2, 0.1]       # `n` coins of given denominations
    target = 0.5               # total change required
 
    # create a dictionary to store solutions to subproblems
    lookup = {}
 
    print('The total number of ways to get the desired change is',
        count(S, len(S) - 1, target, lookup))