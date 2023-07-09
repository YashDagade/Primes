import sympy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os


def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def check_divisibility(numbers, prime):
    divisible = {n for n in numbers if n % prime == 0}
    not_divisible = numbers - divisible
    return divisible, not_divisible

def test_theory(limit, input_num):
    primes = list(sympy.primerange(2, input_num+1))
    numbers = set(range(2, limit + 1))
    divisibility_dict = {}

    for p in tqdm(primes):
        print(f"\nChecking divisibility by {p}...")
        divisible, numbers = check_divisibility(numbers, p)
        divisibility_dict[p] = sorted(list(divisible))
        print(f"Found {len(divisible)} numbers divisible by {p}")

    print(f"\nOut of the first {limit} numbers:")
    for p, divisible_numbers in divisibility_dict.items():
        print(f"Of the remaining, {len(divisible_numbers)} are not prime because they're divisible by {p}")
    print(f"Of the remaining, {len(numbers)} are potential primes (not divisible by any prime up to {input_num})")
    
    return divisibility_dict

def list_all_primes(input_num, divisibility_dict):
    for p in divisibility_dict.keys():
        if p <= input_num:
            print(f"\nThe first 10 individual numbers that are only divisible by {p} are:")
            print(divisibility_dict[p][:10], "...")

def addition(primes):
    remaining_ratio = 1
    ratio_dict = {}

    for prime in primes:
        addition_ratio = remaining_ratio / prime
        ratio_dict[prime] = addition_ratio
        remaining_ratio -= addition_ratio

    return ratio_dict, remaining_ratio

limit = int(input("Enter the upper limit of numbers to check (e.g., 10000): "))
input_num = int(input("Enter the upper limit for primes (e.g., 7): "))

divisibility_dict = test_theory(limit, input_num)

list_all_primes(input_num, divisibility_dict)

primes = list(sympy.primerange(2, input_num+1))

addition_ratios, remaining_ratio = addition(primes)

print("\nAddition Ratios:")
for prime, ratio in addition_ratios.items():
    print(f"Addition for {prime}: {ratio}")

print(f"\nEstimated proportion of primes: {remaining_ratio}")

df = pd.DataFrame(list(addition_ratios.items()), columns=['Prime', 'Ratio'])
df.to_csv(f"{limit}_{input_num}.csv", index=False)

xdata = np.array(primes)
ydata = np.array([addition_ratios[prime] for prime in primes])

popt, pcov = curve_fit(func, xdata, ydata)

plt.figure(figsize=(10,6))
plt.scatter(xdata, ydata, label='data')
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('Prime')
plt.ylabel('Addition Ratio')
plt.legend()
plt.show()


df = pd.DataFrame(list(addition_ratios.items()), columns=['Prime', 'Ratio'])

# Specify the directory
directory = r"C:\Users\Theya\OneDrive - MNSCU\Desktop\Primes\Primes"
filename = f"{limit}_{input_num}.csv"
filepath = os.path.join(directory, filename)  # Combines the directory and filename

df.to_csv(filepath, index=False)

print(f"Data exported to {filepath}")

from scipy import integrate

# Calculate the integral using the trapezoidal rule
integral = integrate.trapz(ydata, xdata)

print(f"The integral of the data is: {integral}")

# Generate data for the fit curve
x_fit = np.linspace(xdata[0], xdata[-1], 1000)
y_fit = func(x_fit, *popt)

plt.figure(figsize=(10,6))
plt.scatter(xdata, ydata, label='data')
plt.plot(x_fit, y_fit, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# Fill the area under the curve
plt.fill_between(x_fit, y_fit, color='gray', alpha=0.5)

plt.xlabel('Prime')
plt.ylabel('Addition Ratio')
plt.legend()
plt.show()
