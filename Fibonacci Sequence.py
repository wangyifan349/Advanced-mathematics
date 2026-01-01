import math
import random
import statistics
import numpy as np

# 1. Fibonacci Sequence (斐波那契数列)
def fibonacci_list(n):
    """Generate a list containing the first n Fibonacci numbers (iterative)."""
    sequence = []
    a = 0
    b = 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence

print("Fibonacci sequence (n=10):", fibonacci_list(10))

# 2. Factorial (阶乘)
def factorial(n):
    """Calculate the factorial of n."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print("7! =", factorial(7))

# 3. Permutations and Combinations (排列组合)
def permutations(n, k):
    """Return the number of k-permutations of n (P(n, k))."""
    result = 1
    for i in range(n, n - k, -1):
        result *= i
    return result

def combinations(n, k):
    """Return the number of k-combinations of n (C(n, k))."""
    return factorial(n) // (factorial(k) * factorial(n - k))

print("P(5,3) permutations:", permutations(5, 3))
print("C(5,3) combinations:", combinations(5, 3))

# 4. Random Sampling
data_list = [1, 2, 3, 4, 5, 6]
random_element = random.choice(data_list)
print("Random choice from list:", random_element)

random_sample = []
sample_size = 3
data_copy = data_list[:]
for _ in range(sample_size):
    element = random.choice(data_copy)
    random_sample.append(element)
    data_copy.remove(element)
print("Random sample of 3:", random_sample)

# 5. Statistics (mean, median, standard deviation)
number_list = [2, 5, 1, 8, 7, 3]

def mean(numbers):
    """Calculate the mean of a list of numbers."""
    total = 0
    count = 0
    for element in numbers:
        total += element
        count += 1
    return total / count if count else 0

def median(numbers):
    """Calculate the median of a list of numbers."""
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_numbers[n // 2]
    else:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2

def standard_deviation(numbers):
    """Calculate the standard deviation of a list of numbers."""
    avg = mean(numbers)
    squared_diffs = []
    for element in numbers:
        squared_diffs.append((element - avg) ** 2)
    variance = mean(squared_diffs)
    return math.sqrt(variance)

print("Mean:", mean(number_list))
print("Median:", median(number_list))
print("Standard deviation:", standard_deviation(number_list))

# 6. Range Sum and Product
def sum_range(start, end):
    """Return the sum of integers from start to end (inclusive)."""
    result = 0
    for i in range(start, end + 1):
        result += i
    return result

def product_range(start, end):
    """Return the product of integers from start to end (inclusive)."""
    result = 1
    for i in range(start, end + 1):
        result *= i
    return result

print("Sum 1 to 100:", sum_range(1, 100))
print("Product 1 to 5:", product_range(1, 5))

# 7. Euclidean Distance (欧氏距离)
def euclidean_distance(point_a, point_b):
    """Calculate the Euclidean distance between two points."""
    squared_differences = []
    for value_a, value_b in zip(point_a, point_b):
        squared_differences.append((value_a - value_b) ** 2)
    return math.sqrt(sum(squared_differences))

print("Distance between (1,2) and (3,4):", euclidean_distance((1, 2), (3, 4)))

# 8. Prime Numbers with For-loops (素数判断和输出)
def is_prime(number):
    """Check if a number is prime."""
    if number < 2:
        return False
    for possible_factor in range(2, int(math.sqrt(number)) + 1):
        if number % possible_factor == 0:
            return False
    return True

def primes_up_to(limit):
    """Return a list of prime numbers up to 'limit' (inclusive)."""
    primes = []
    for number in range(2, limit + 1):
        if is_prime(number):
            primes.append(number)
    return primes

print("Primes up to 20:", primes_up_to(20))

# 9. GCD and LCM (最大公约数和最小公倍数)
def greatest_common_divisor(a, b):
    """Calculate the greatest common divisor (GCD) of a and b."""
    while b:
        a, b = b, a % b
    return a

def least_common_multiple(a, b):
    """Calculate the least common multiple (LCM) of a and b."""
    return a * b // greatest_common_divisor(a, b)

print("GCD(24, 36):", greatest_common_divisor(24, 36))
print("LCM(24, 36):", least_common_multiple(24, 36))

# 10. Sum of Powers (幂次和)
def sum_of_powers(n, power):
    """Return the sum of i**power from i=1 to n."""
    result = 0
    for i in range(1, n + 1):
        result += i ** power
    return result

print("Sum of squares 1~5:", sum_of_powers(5, 2))
