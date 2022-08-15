import string
import math 
import multiprocessing as mp
import itertools

"""
for a new coin machine, which accepts any amount of notes or coins, and dispensing equal value of coins for change.
computes solutions of coin combination

BUT THE EXAMPLE ON THE PDF IS INCORRECT. if the above should be created; then technically 50 * 1 pence also is a valid solution, similarly combinations of 0.02 etc.
This could take a long time to calculate
"""

# unlimited number of coins in the machine
class CoinChange():
    def __init__(self):
        self.coins_list = [2., 1., 0.50, 0.20, 0.10, 0.05, 0.02, 0.01] # [0.5, 0.20, 0.10] #0
        self.sols = []
    
    def calculate_all_combinations(self, sum_input, p, coins):
        if sum_input == 0:
            #global self.sols
            self.sols.append(p)
        # This simply requires Combinations to be calculated via Brute Force (Recursion)
        if sum_input == 0: return 1 
        # return 0 if total becomes negative
        if sum_input < 0: return 0
        total_solutions = 0
        
        # do for each coin
        for c in self.coins_list:
            if sum_input - c >= 0:
                self.calculate_all_combinations(sum_input-c,p+[c],coins=coins)    
        #print('all combinations completed')

    def odd_count(self):
        c=0
        for i in self.sols:
            if len(i)%2!=0:
                c+=1
            else:
                pass
        return c


    def calculate_change(self, input):
        """args:
        input = input string '£{pound}-{pence}'
        output = tuple, where [0] = odd_count = is the total count of solutions with odd coin count, 
                                [1] = self.sols = is the actual solutions list

        """
        split_string = input.split("-")
        if len(split_string[0])>0:
            pounds = float(split_string[0].replace("£",""))
        else:
            pounds = 0.0
        
        if len(split_string[1])>0:
            pence = float('0.{}'.format(split_string[1]))
        else:
            pence = 0.0
        
        # get the total sum to breakdown
        print('input pounds: ',pounds, 'input pence: ', pence)
        sum_input = pounds + pence
        # define all solutions;
        self.calculate_all_combinations(sum_input, p=[], coins = self.coins_list)
        odd_count = self.odd_count()

        return odd_count, self.sols

if __name__=='__main__':
    print("Number of processors: ", mp.cpu_count())
    print("running code via multi processing")

    coin_change = CoinChange() 
    input_string = input("enter input value in the form: '£{pounds}-{pence}':")
    output = coin_change.calculate_change(input_string)
    print("--------------")
    print("Odd coin count combinations:", output[0])
    print("--------------")
    print("Total no. of solutions:", len(output[1]))
    print("--------------")
    print("Solutions: \n", output[1])


    
