import math
real_value = math.sqrt( (1+math.sqrt(5))/2 + 2 ) - (1+math.sqrt(5))/2
print("real_value:",real_value)

term2 = math.exp(-2 * math.pi)

a = math.exp((-2 * math.pi) / 5)
def golden_ratio(n):
    if n > 1:
        return 1 + term2/(golden_ratio(n-1) )
    else:
        return term2


result = math.exp( (-2 * math.pi)/5 )  / golden_ratio(500)



print("The approximate value of the golden ratio formula is:", result)
