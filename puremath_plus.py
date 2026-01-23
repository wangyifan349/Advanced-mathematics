# ========================= puremath_plus.py ===========================
# Dependency-free math package: undergrad/grad-level, all-English naming

EPSILON = 1e-12                                   # Numerical tolerance

# 1. Elementary mathematics
class BasicMath:
    PI = 3.141592653589793
    TAU = 2 * PI
    E = 2.718281828459045

    @staticmethod
    def add(a, b): return a + b                   # Addition
    @staticmethod
    def subtract(a, b): return a - b              # Subtraction
    @staticmethod
    def multiply(a, b): return a * b              # Multiplication
    @staticmethod
    def divide(a, b):                             # Division
        if abs(b) < EPSILON:
            raise ZeroDivisionError("Division by zero")
        return a / b

    @staticmethod
    def power(x, n):                              # Integer exponentiation
        result = 1.0
        is_positive = n >= 0
        for _ in range(abs(int(n))):
            result *= x
        return result if is_positive else 1.0 / result

    @staticmethod
    def factorial(n: int):                        # Factorial
        if n < 0:
            raise ValueError("Negative factorial")
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    @staticmethod
    def sqrt(x, tol=EPSILON):                     # Square root (Newton)
        if x < 0:
            raise ValueError("sqrt of negative")
        guess = x if x > 1 else 1.0
        while abs(guess * guess - x) > tol:
            guess = 0.5 * (guess + x / guess)
        return guess

    @staticmethod
    def exp(x, terms=30):                         # Exponential function
        term = 1.0
        summation = 1.0
        for n in range(1, terms):
            term *= x / n
            summation += term
        return summation

    @staticmethod
    def ln(x, tol=EPSILON):                       # Natural logarithm
        if x <= 0:
            raise ValueError("ln domain x>0")
        y = x - 1.0
        for _ in range(100):
            y_new = y - (BasicMath.exp(y) - x) / BasicMath.exp(y)
            if abs(y_new - y) < tol:
                break
            y = y_new
        return y

    @staticmethod
    def sin(x, terms=12):                         # Sine (series)
        x %= BasicMath.TAU
        term = x
        result = x
        sign = -1
        for n in range(1, terms):
            term *= x * x / ((2 * n) * (2 * n + 1))
            result += sign * term
            sign *= -1
        return result

    @staticmethod
    def cos(x, terms=12):                         # Cosine (series)
        x %= BasicMath.TAU
        term = 1.0
        result = 1.0
        sign = -1
        for n in range(1, terms):
            term *= x * x / ((2 * n - 1) * (2 * n))
            result += sign * term
            sign *= -1
        return result

    @staticmethod
    def tan(x):                                   # Tangent
        return BasicMath.sin(x) / BasicMath.cos(x)

    @staticmethod
    def arctan(x, terms=15):                      # Arctangent (series)
        if abs(x) > 1:
            return (BasicMath.PI / 2 - BasicMath.arctan(1 / x, terms)) if x > 0 \
                else (-BasicMath.PI / 2 - BasicMath.arctan(1 / x, terms))
        result = 0
        sign = 1
        power = x
        for n in range(1, 2 * terms, 2):
            result += sign * power / n
            power *= x * x
            sign *= -1
        return result

# 2. Vector algebra
class Vector:
    def __init__(self, values):                    # Constructor
        self.values = [float(v) for v in values]
    def __len__(self): return len(self.values)     # Length
    def __getitem__(self, i): return self.values[i] # Indexing
    def __repr__(self): return f"Vector({self.values})"
    def __add__(self, other):                      # Vector addition
        return Vector([a + b for a, b in zip(self.values, other.values)])
    def __sub__(self, other):                      # Vector subtraction
        return Vector([a - b for a, b in zip(self.values, other.values)])
    def __mul__(self, other):                      # Dot or scalar product
        if isinstance(other, Vector):
            return sum(a * b for a, b in zip(self.values, other.values))
        else:
            return Vector([a * other for a in self.values])
    def norm(self):                                # Euclidean norm
        return BasicMath.sqrt(sum(x * x for x in self.values))
    def cross(self, other):                        # Cross product (3D)
        if len(self) != 3 or len(other) != 3:
            raise ValueError("3-D only")
        a, b = self.values, other.values
        return Vector([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ])
    def to_list(self): return self.values[:]        # Convert to list

# 3. Matrix algebra (Gauss / LU / rank ...)
class Matrix:
    def __init__(self, data):                      # Constructor (rectangular required)
        if len({len(row) for row in data}) != 1:
            raise ValueError("ragged")
        self.data = [list(map(float, row)) for row in data]
    def shape(self):                               # Returns (row_count, col_count)
        return len(self.data), len(self.data[0])
    def __getitem__(self, i): return self.data[i]  # Indexing
    def __repr__(self): return f"Matrix({self.data})"
    def copy(self): return Matrix([row[:] for row in self.data]) # Deep copy
    def to_list(self): return [row[:] for row in self.data]      # Convert to list

    def __add__(self, other):                      # Matrix addition
        rows, cols = self.shape()
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(cols)] for i in range(rows)])
    def __mul__(self, other):                      # Matrix multiplication
        if isinstance(other, Matrix):
            r1, c1 = self.shape()
            r2, c2 = other.shape()
            if c1 != r2:
                raise ValueError("size mismatch")
            return Matrix([[sum(self.data[i][k] * other.data[k][j] for k in range(c1))
                            for j in range(c2)] for i in range(r1)])
        return Matrix([[element * other for element in row] for row in self.data])
    def transpose(self):                           # Matrix transpose
        rows, cols = self.shape()
        return Matrix([[self.data[i][j] for i in range(rows)] for j in range(cols)])

    def determinant(self):                         # Determinant (Gauss elimination)
        n, m = self.shape()
        if n != m:
            raise ValueError("square only")
        temp_matrix = [row[:] for row in self.data]
        det = 1.0
        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(temp_matrix[r][i]))
            if abs(temp_matrix[pivot][i]) < EPSILON:
                return 0.0
            if pivot != i:
                temp_matrix[i], temp_matrix[pivot] = temp_matrix[pivot], temp_matrix[i]
                det = -det
            det *= temp_matrix[i][i]
            for j in range(i + 1, n):
                factor = temp_matrix[j][i] / temp_matrix[i][i]
                for k in range(i, n):
                    temp_matrix[j][k] -= factor * temp_matrix[i][k]
        return det

    def inverse(self):                             # Inverse matrix (Gauss-Jordan)
        n, m = self.shape()
        if n != m:
            raise ValueError("square only")
        a = [row[:] for row in self.data]
        identity = [[float(i == j) for j in range(n)] for i in range(n)]
        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
            if abs(a[pivot][i]) < EPSILON:
                raise ValueError("singular")
            if pivot != i:
                a[i], a[pivot] = a[pivot], a[i]
                identity[i], identity[pivot] = identity[pivot], identity[i]
            pv = a[i][i]
            for j in range(n):
                a[i][j] /= pv
                identity[i][j] /= pv
            for r in range(n):
                if r == i:
                    continue
                factor = a[r][i]
                for j in range(n):
                    a[r][j] -= factor * a[i][j]
                    identity[r][j] -= factor * identity[i][j]
        return Matrix(identity)

    def rank(self):                               # Matrix rank
        temp_matrix = [row[:] for row in self.data]
        rows, cols = self.shape()
        row_idx = 0
        rank = 0
        for col in range(cols):
            pivot = None
            for i in range(row_idx, rows):
                if abs(temp_matrix[i][col]) > EPSILON:
                    pivot = i
                    break
            if pivot is None:
                continue
            temp_matrix[row_idx], temp_matrix[pivot] = temp_matrix[pivot], temp_matrix[row_idx]
            pv = temp_matrix[row_idx][col]
            for j in range(col, cols):
                temp_matrix[row_idx][j] /= pv
            for i in range(rows):
                if i != row_idx:
                    factor = temp_matrix[i][col]
                    for j in range(col, cols):
                        temp_matrix[i][j] -= factor * temp_matrix[row_idx][j]
            row_idx += 1
            rank += 1
            if row_idx == rows:
                break
        return rank

    def lu_decomposition(self):                   # Doolittle LU decomposition
        n, m = self.shape()
        if n != m:
            raise ValueError("square only")
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(i, n):
                U[i][k] = self.data[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
            L[i][i] = 1.0
            for k in range(i + 1, n):
                if abs(U[i][i]) < EPSILON:
                    raise ValueError("zero pivot (LU)")
                L[k][i] = (self.data[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]
        return Matrix(L), Matrix(U)

    def solve(self, b):                           # Solve linear system Ax=b
        if not isinstance(b[0], list):
            b = [[v] for v in b]
        aug = [row + rhs for row, rhs in zip(self.copy().data, b)]
        n = len(aug)
        m = len(aug[0])
        col = 0
        for row in range(n):
            pivot = None
            for r in range(row, n):
                if abs(aug[r][col]) > EPSILON:
                    pivot = r
                    break
            while pivot is None:
                col += 1
                if col >= m - len(b[0]):
                    raise ValueError("no unique solution")
                for r in range(row, n):
                    if abs(aug[r][col]) > EPSILON:
                        pivot = r
                        break
            if pivot != row:
                aug[row], aug[pivot] = aug[pivot], aug[row]
            pv = aug[row][col]
            for c in range(col, m):
                aug[row][c] /= pv
            for r in range(n):
                if r == row:
                    continue
                factor = aug[r][col]
                for c in range(col, m):
                    aug[r][c] -= factor * aug[row][c]
            col += 1
        sol_columns = m - self.shape()[1]
        result = [[aug[r][self.shape()[1] + c] for c in range(sol_columns)] for r in range(self.shape()[0])]
        return [r[0] for r in result] if sol_columns == 1 else result

# 4. Numerical calculus
class Calculus:
    @staticmethod
    def derivative(f, x, h=1e-8):                # First derivative
        return (f(x + h) - f(x - h)) / (2 * h)
    @staticmethod
    def nth_derivative(f, x, n, h=1e-5):         # n-th derivative
        return f(x) if n == 0 else (Calculus.nth_derivative(f, x + h, n - 1, h) -
                                    Calculus.nth_derivative(f, x - h, n - 1, h)) / (2 * h)
    @staticmethod
    def integral(f, a, b, n=2000):               # Definite integral (trapezoid)
        h = (b - a) / n
        summation = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            summation += f(a + i * h)
        return summation * h
    @staticmethod
    def simpson(f, a, b, n=500):                 # Simpson's rule
        if n % 2:
            n += 1
        h = (b - a) / n
        summation = f(a) + f(b)
        for i in range(1, n, 2):
            summation += 4 * f(a + i * h)
        for i in range(2, n, 2):
            summation += 2 * f(a + i * h)
        return summation * h / 3
    @staticmethod
    def partial_derivative(f, point, idx, h=1e-6): # Partial derivative
        p1 = list(point)
        p2 = list(point)
        p1[idx] += h
        p2[idx] -= h
        return (f(*p1) - f(*p2)) / (2 * h)
    @staticmethod
    def taylor_series(f, a, order=5):            # Taylor series coefficients and polynomial
        coefficients = [Calculus.nth_derivative(f, a, k) / BasicMath.factorial(k)
                        for k in range(order + 1)]
        def polynomial(x):
            s = 0.0
            for k, c in enumerate(coefficients):
                s += c * (x - a) ** k
            return s
        return coefficients, polynomial
    @staticmethod
    def double_integral(f, ax, bx, ay, by, nx=60, ny=60): # 2D integral
        hx = (bx - ax) / nx
        hy = (by - ay) / ny
        summation = 0.0
        for i in range(nx):
            for j in range(ny):
                summation += f(ax + (i + 0.5) * hx, ay + (j + 0.5) * hy)
        return summation * hx * hy

# 5. Polynomial algebra
class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = list(map(float, coefficients))
    def degree(self):
        return len(self.coefficients) - 1
    def __call__(self, x):
        result = 0.0
        for coef in reversed(self.coefficients):
            result = result * x + coef
        return result
    def derivative(self):
        if self.degree() == 0:
            return Polynomial([0])
        return Polynomial([(i) * self.coefficients[i] for i in range(1, len(self.coefficients))])
    def integral(self, constant=0.0):
        return Polynomial([constant] + [self.coefficients[i] / (i + 1) for i in range(len(self.coefficients))])
    def __add__(self, other):
        m = max(len(self.coefficients), len(other.coefficients))
        return Polynomial([(self.coefficients[i] if i < len(self.coefficients) else 0) +
                           (other.coefficients[i] if i < len(other.coefficients) else 0) for i in range(m)])
    def __mul__(self, other):
        res = [0.0] * (len(self.coefficients) + len(other.coefficients) - 1)
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                res[i + j] += a * b
        return Polynomial(res)
    def root_newton(self, x0, tol=1e-10, max_iter=100):
        p = self
        dp = self.derivative()
        x = x0
        for _ in range(max_iter):
            fx = p(x)
            dfx = dp(x)
            if abs(dfx) < EPSILON:
                break
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        raise RuntimeError("Newton failed")

# 6. Statistics
class Statistics:
    @staticmethod
    def mean(data):
        return sum(data) / len(data)
    @staticmethod
    def median(data):
        s = sorted(data)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    @staticmethod
    def variance(data, population=True):
        mu = Statistics.mean(data)
        n = len(data)
        return sum((x - mu) ** 2 for x in data) / (n if population else n - 1)
    @staticmethod
    def std_dev(data, population=True):
        return BasicMath.sqrt(Statistics.variance(data, population))
    @staticmethod
    def covariance(x, y):
        mu_x = Statistics.mean(x)
        mu_y = Statistics.mean(y)
        n = len(x)
        return sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / n
    @staticmethod
    def correlation(x, y):
        return Statistics.covariance(x, y) / (Statistics.std_dev(x) * Statistics.std_dev(y))

# 7. Number theory / combinatorics
class NumberTheory:
    @staticmethod
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
    @staticmethod
    def lcm(a, b):
        return abs(a * b) // NumberTheory.gcd(a, b)
    @staticmethod
    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = NumberTheory.extended_gcd(b, a % b)
        return g, y1, x1 - (a // b) * y1
    @staticmethod
    def mod_inverse(a, m):
        g, x, _ = NumberTheory.extended_gcd(a, m)
        if g != 1:
            raise ValueError("no inverse")
        return x % m
    @staticmethod
    def pow_mod(base, exp, mod):
        res = 1 % mod
        b = base % mod
        while exp:
            if exp & 1:
                res = (res * b) % mod
            b = (b * b) % mod
            exp //= 2
        return res
    @staticmethod
    def is_prime(n):                              # Miller-Rabin deterministic test
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in (2, 7, 61):
            if a % n == 0:
                continue
            x = NumberTheory.pow_mod(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    @staticmethod
    def binomial(n, k):
        if k < 0 or k > n:
            return 0
        k = min(k, n - k)
        result = 1
        for i in range(1, k + 1):
            result = result * (n - k + i) // i
        return result

# 8. Special functions (approximations)
class SpecialFunctions:
    @staticmethod
    def gamma(x):                                  # Gamma function (Lanczos)
        if x < 0.5:
            return BasicMath.PI / (BasicMath.sin(BasicMath.PI * x) * SpecialFunctions.gamma(1 - x))
        coeff = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                 771.32342877765313, -176.61502916214059, 12.507343278686905,
                 -0.13857109526572012, 9.9843695780195716e-6,
                 1.5056327351493116e-7]
        g = 7
        y = coeff[0]
        t = x + g + 0.5
        for i in range(1, len(coeff)):
            y += coeff[i] / (x + i - 1)
        sqrt_two_pi = 2.5066282746310007
        return sqrt_two_pi * (t ** (x + 0.5)) * BasicMath.exp(-t) * y

    @staticmethod
    def beta(x, y):                                # Beta function
        return SpecialFunctions.gamma(x) * SpecialFunctions.gamma(y) / SpecialFunctions.gamma(x + y)
    @staticmethod
    def erf(z, terms=25):                          # Error function
        summation = 0.0
        for n in range(terms):
            summation += ((-1) ** n) * z ** (2 * n + 1) / (BasicMath.factorial(n) * (2 * n + 1))
        return 2 * summation / BasicMath.sqrt(BasicMath.PI)

# 9. Advanced linear algebra
class LinearAlgebraAdv:
    @staticmethod
    def qr_decomposition(A: Matrix):                # QR decomposition (Gram-Schmidt)
        m, n = A.shape()
        Q = [[0.0] * n for _ in range(m)]
        R = [[0.0] * n for _ in range(n)]
        a_cols = [[A[i][j] for i in range(m)] for j in range(n)]
        q_cols = []
        for j in range(n):
            v = a_cols[j][:]
            for k in range(j):
                r = sum(v_i * q_i for v_i, q_i in zip(v, [row[k] for row in Q]))
                R[k][j] = r
                for i in range(m):
                    v[i] -= r * Q[i][k]
            norm = BasicMath.sqrt(sum(v_i * v_i for v_i in v))
            if norm < EPSILON:
                continue
            R[j][j] = norm
            for i in range(m):
                Q[i][j] = v[i] / norm
            q_cols.append([Q[i][j] for i in range(m)])
        return Matrix(Q), Matrix(R)

    @staticmethod
    def power_iteration(A: Matrix, iter_max=1000, tol=1e-9): # Largest eigenvalue/vector
        n, _ = A.shape()
        b = [1.0] * n
        for _ in range(iter_max):
            Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
            norm = BasicMath.sqrt(sum(x * x for x in Ab))
            b_next = [x / norm for x in Ab]
            if max(abs(b_next[i] - b[i]) for i in range(n)) < tol:
                eigenvalue = sum(b_next[i] * Ab[i] for i in range(n))
                return eigenvalue, b_next
            b = b_next
        raise RuntimeError("no convergence")

# 10. Optimization
class Optimization:
    @staticmethod
    def newton_1d(f, df, x0, tol=1e-10, max_iter=100): # Newton–Raphson (1D)
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            dfx = df(x)
            if abs(dfx) < EPSILON:
                break
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        raise RuntimeError("Newton failed")

    @staticmethod
    def gradient_descent(f, grad, x0, lr=1e-2, tol=1e-6, max_iter=10000): # Gradient descent
        x = list(x0)
        for _ in range(max_iter):
            g = grad(*x)
            x_new = [xi - lr * gi for xi, gi in zip(x, g)]
            if max(abs(a - b) for a, b in zip(x_new, x)) < tol:
                return x_new
            x = x_new
        raise RuntimeError("no converge")

# 11. ODE solver (Euler and Runge-Kutta)
class ODESolver:
    @staticmethod
    def euler(f, y0, t0, t1, h):                   # Euler method
        t = t0
        y = y0
        while t < t1 - EPSILON:
            y = [yi + h * fi for yi, fi in zip(y, f(t, y))]
            t += h
        return y
    @staticmethod
    def rk4(f, y0, t0, t1, h):                     # Runge-Kutta 4th order
        t = t0
        y = y0
        while t < t1 - EPSILON:
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, [yi + 0.5 * h * k1i for yi, k1i in zip(y, k1)])
            k3 = f(t + 0.5 * h, [yi + 0.5 * h * k2i for yi, k2i in zip(y, k2)])
            k4 = f(t + h, [yi + h * k3i for yi, k3i in zip(y, k3)])
            y = [yi + h / 6 * (k1i + 2 * k2i + 2 * k3i + k4i)
                 for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)]
            t += h
        return y

# 12. Fast Fourier Transform (Cooley-Tukey, radix-2)
class FFT:
    @staticmethod
    def _fft(a, invert):                        # Internal recursive FFT utility
        n = len(a)
        if n == 1:
            return a
        even = FFT._fft(a[0::2], invert)
        odd = FFT._fft(a[1::2], invert)
        angle = 2 * BasicMath.PI / n * (-1 if invert else 1)
        w = 1 + 0j
        wn = complex(BasicMath.cos(angle), BasicMath.sin(angle))
        y = [0] * n
        for k in range(n // 2):
            y[k] = even[k] + w * odd[k]
            y[k + n // 2] = even[k] - w * odd[k]
            w *= wn
        return y
    @staticmethod
    def fft(seq):                              # FFT (n must be 2^k)
        n = len(seq)
        if n & (n - 1):
            raise ValueError("length must be power of 2")
        return FFT._fft([complex(x) for x in seq], False)
    @staticmethod
    def ifft(seq):                             # Inverse FFT
        n = len(seq)
        res = FFT._fft([complex(x) for x in seq], True)
        return [x / n for x in res]

# 13. Self-test (if this file is run directly)
if __name__ == "__main__":
    print("BasicMath  sin(pi/2) =", BasicMath.sin(BasicMath.PI / 2))  # ~1.0
    print("Matrix det  :", Matrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]]).determinant())
    p = Polynomial([-2, 0, 1])
    print("sqrt(2) via Newton =", p.root_newton(1.5))                 # ~1.4142
    data = [1, 2, 3, 4, 5]
    print("mean/var:", Statistics.mean(data), Statistics.variance(data))
    print("is 2147483647 prime? ", NumberTheory.is_prime(2147483647)) # True
    print("Gamma(5) should be 24 ->", SpecialFunctions.gamma(5))      # ~24.0
    def ode_func(t, y): return [y[0]]
    print("rk4 exp(1)≈", ODESolver.rk4(ode_func, [1.0], 0, 1, 0.01)[0])
    samples = [0, 1, 0, -1] * 4
    print("FFT len", len(samples), "->", FFT.fft(samples)[:4])
