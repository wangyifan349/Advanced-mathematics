import sympy as sp

# 定义符号变量
x, y, t = sp.symbols('x y t')
n = sp.symbols('n')

# 微积分相关
print("--- 微积分操作 ---")
# 极限
print("极限1:", sp.limit(sp.sin(x)/x, x, 0))
print("极限2:", sp.limit(1/x, x, sp.oo))
print("极限3:", sp.limit((1-sp.cos(x))/x**2, x, 0))
# 导数
print("导数1:", sp.diff(x**2 + 2*x + 1, x))
print("导数2:", sp.diff(sp.exp(2*x), x))
print("导数3:", sp.diff(sp.log(x)/x, x))
# 定积分与不定积分
print("不定积分1:", sp.integrate(sp.cos(x), x))
print("不定积分2:", sp.integrate(sp.exp(x**2), x))
print("定积分1:", sp.integrate(sp.sin(x), (x, 0, sp.pi)))
print("定积分2:", sp.integrate(sp.log(x), (x, 1, sp.E)))
# 多重积分
print("二重积分:", sp.integrate(sp.integrate(x + y, (x, 0, 1)), (y, 0, 1)))

# 高等数学操作
print("\n--- 高等数学操作 ---")
# 泰勒展开
print("sin(x)泰勒展开:", sp.series(sp.sin(x), x, 0, 6))
# 偏导数
f = x**2 * y + y**2 * x
print("偏导x:", sp.diff(f, x))
print("偏导y:", sp.diff(f, y))
# 二阶导数
print("exp(x)二阶导:", sp.diff(sp.exp(x), x, 2))
# 级数和
print("级数sum1/n**2:", sp.summation(1/n**2, (n,1,sp.oo)))
# 微分方程求解
y_func = sp.Function('y')
ode = sp.Eq(y_func(x).diff(x) + y_func(x), 0)
print("微分方程解:", sp.dsolve(ode, y_func(x), ics={y_func(0): 1}))
# 拉普拉斯变换
print("拉普拉斯变换sin(3t):", sp.laplace_transform(sp.sin(3*t), t, s=True))
# 傅里叶变换
print("傅里叶变换exp(-x**2):", sp.fourier_transform(sp.exp(-x**2), x, y))
# 矩阵与行列式
A = sp.Matrix([[1, 2], [3, 4]])
print("矩阵行列式:", A.det())
print("矩阵特征值:", A.eigenvals())
# 定积分变量替换（实际用常规解法）
print("定积分 sqrt(1-x**2):", sp.integrate(sp.sqrt(1-x**2), (x, 0, 1)))

# 线性代数操作
print("\n--- 线性代数操作 ---")
# 高斯消元法（解线性方程）
eq1 = sp.Eq(2*x + y, 8)
eq2 = sp.Eq(3*x + 2*y, 13)
print("高斯消元法解方程组:", sp.solve([eq1, eq2], (x, y)))
# 矩阵特征分解
A = sp.Matrix([[4, 1], [2, 3]])
print("矩阵A特征值:", A.eigenvals())
print("矩阵A特征向量:", A.eigenvects())
# 对角化
P, D = A.diagonalize()
print("对角化P:\n", P)
print("对角化D:\n", D)
# QR分解
Q, R = A.QRdecomposition()
print("QR分解Q:\n", Q)
print("QR分解R:\n", R)
# 谱相似度（特征值比较）
B = sp.Matrix([[4, 2], [1, 3]])
eigen_A = set(A.eigenvals().keys())
eigen_B = set(B.eigenvals().keys())
print("A与B特征值是否相同:", eigen_A == eigen_B)
print("A特征值:", eigen_A)
print("B特征值:", eigen_B)
# 奇异值分解SVD
A_num = sp.Matrix([[1, 2], [3, 4]])
U, S, V = A_num.singular_value_decomposition()
print("SVD U:\n", U)
print("SVD S:\n", S)
print("SVD V:\n", V)
# 逆矩阵与秩
print("逆矩阵:\n", A.inv())
print("矩阵秩:", A.rank())
# 矩阵乘法
B2 = sp.Matrix([[1, 0], [0, 1]])
print("矩阵乘法:\n", A * B2)
# 向量点积与叉积
v1 = sp.Matrix([1, 2, 3])
v2 = sp.Matrix([4, 5, 6])
print("点积:", v1.dot(v2))
print("叉积:", v1.cross(v2))



import sympy as sp

# 微分方程相关
print("--- 一阶常微分方程 ---")
x, y = sp.symbols('x y')
C = sp.symbols('C')
f = sp.Function('f')

# 一阶变量可分离方程：dy/dx = x*y
ode1 = sp.Eq(f(x).diff(x), x * f(x))
sol1 = sp.dsolve(ode1)
print("一阶变量分离方程 dy/dx = x*y 的通解：", sol1)

# 一阶线性方程：dy/dx + y = 2
ode2 = sp.Eq(f(x).diff(x) + f(x), 2)
sol2 = sp.dsolve(ode2)
print("一阶线性方程 dy/dx + y = 2 的通解：", sol2)

# 一阶常系数齐次方程：dy/dx + 3y = 0，y(0)=2
ode3 = sp.Eq(f(x).diff(x) + 3 * f(x), 0)
sol3 = sp.dsolve(ode3, ics={f(0): 2})
print("初值问题解 dy/dx + 3y = 0, y(0)=2：", sol3)

# 二阶常系数齐次方程：y'' - 2y' + y = 0
ode4 = sp.Eq(f(x).diff(x, 2) - 2*f(x).diff(x) + f(x), 0)
sol4 = sp.dsolve(ode4)
print("二阶常系数齐次方程解：", sol4)

# 拉普拉斯方程 PDE: φ_xx + φ_yy = 0
phi = sp.Function('phi')
X, Y = sp.symbols('X Y')
pde1 = sp.Eq(sp.diff(phi(X,Y), X, 2) + sp.diff(phi(X,Y), Y, 2), 0)
print("拉普拉斯方程 PDE 结构：", pde1)

# 偏微分方程的通用框架（不能直接自动求解，实际需要给定特定边界条件）

# 级数解法示例（泰勒展开作为ODE的近似）
fexpr = sp.Function('f')(x)
ode5 = fexpr.diff(x) - x*fexpr
sol5_series = sp.series(sp.dsolve(ode5), x, 0, 6)
print("级数近似解（常微分方程）：", sol5_series)

# 常见的高等数学补充
print("\n--- 其他高等数学操作 ---")

# 卷积
t, tau = sp.symbols('t tau')
g = sp.sin(t)
h = sp.cos(t)
convolution = sp.integrate(g.subs(t, tau) * h.subs(t, t - tau), (tau, 0, t))
print("sin(t)与cos(t)的卷积：", convolution)

# 狄拉克δ函数积分性质
delta_x = sp.DiracDelta(x)
print("∫f(x)δ(x-a)dx = f(a)：", sp.integrate(sp.sin(x)*sp.DiracDelta(x-sp.pi/2), (x, -sp.oo, sp.oo)))

# 分段函数
piecewise = sp.Piecewise((x, x < 1), (x**2, x >= 1))
print("分段函数表达式:", piecewise)

# 黑塞矩阵（Hessian，对多元函数二阶偏导矩阵）
u, v = sp.symbols('u v')
F = u**2 * v + sp.sin(u*v)
hess = sp.hessian(F, (u, v))
print("二元函数黑塞矩阵:\n", hess)

# 构造jacobi矩阵（雅可比矩阵）
f1 = u**2 + v
f2 = sp.sin(u * v)
J = sp.Matrix([f1, f2]).jacobian([u, v])
print("jacobi矩阵:\n", J)

# 格林公式类型积分（实际演算需定域）
print("格林公式类型表达：(∮C Pdx+Qdy = ∬D(Q_x-P_y)dA)")

# 傅里叶级数展开示例
a0, an, bn, L = sp.symbols('a0 an bn L')
fs = sp.fourier_series(sp.Abs(sp.sin(x)), (x, -sp.pi, sp.pi))
print("|sin(x)| 的傅里叶级数展开:\n", fs)

# 多重积分、曲线与曲面积分示意
print("多重积分示例：", sp.integrate(sp.integrate(x*y, (x, 0, 1)), (y, 0, 1)))
print("曲线、曲面积分可用sp.integrate结合参数化变量表达，例如∮_C f(x,y)dx+g(x,y)dy")

# 概率密度/分布函数
mu, sigma = sp.symbols('mu sigma')
normal_dist = 1/(sp.sqrt(2*sp.pi)*sigma) * sp.exp(-(x-mu)**2/(2*sigma**2))
print("正态分布函数表达：", normal_dist)

# 拉普拉斯算子（用于PDE/物理）
print("拉普拉斯算子:", sp.diff(f(x, y), x, 2) + sp.diff(f(x, y), y, 2))





import sympy as sp

x = sp.symbols('x')
f = sp.Function('f')

# 一阶变量可分离：dy/dx = 2*x*y
ode1 = sp.Eq(f(x).diff(x), 2*x*f(x))
sol1 = sp.dsolve(ode1)
print("dy/dx = 2*x*y 的通解：", sol1)

# 一阶变量可分离：dy/dx = -y/x
ode2 = sp.Eq(f(x).diff(x), -f(x)/x)
sol2 = sp.dsolve(ode2)
print("dy/dx = -y/x 的通解：", sol2)

# 一阶线性非齐次：dy/dx + 5y = 4x
ode3 = sp.Eq(f(x).diff(x) + 5*f(x), 4*x)
sol3 = sp.dsolve(ode3)
print("dy/dx + 5y = 4x 的通解：", sol3)

# 一阶线性非齐次：dy/dx - 2y = e^x，y(0)=1
ode4 = sp.Eq(f(x).diff(x) - 2*f(x), sp.exp(x))
sol4 = sp.dsolve(ode4, ics={f(0): 1})
print("dy/dx - 2y = exp(x), y(0)=1 的解：", sol4)

# 二阶常系数齐次：y'' + y = 0
ode5 = sp.Eq(f(x).diff(x, 2) + f(x), 0)
sol5 = sp.dsolve(ode5)
print("y'' + y = 0 的通解：", sol5)

# 二阶常系数非齐次：y'' - y = x
ode6 = sp.Eq(f(x).diff(x, 2) - f(x), x)
sol6 = sp.dsolve(ode6)
print("y'' - y = x 的通解：", sol6)

# 二阶常系数初值问题：y'' + 2y' + y = 0, y(0)=0, y'(0)=1
ode7 = sp.Eq(f(x).diff(x, 2) + 2 * f(x).diff(x) + f(x), 0)
sol7 = sp.dsolve(ode7, ics={f(0): 0, f(x).diff(x).subs(x,0): 1})
print("y'' + 2y' + y = 0, y(0)=0, y'(0)=1 的解：", sol7)

# 二阶欧拉方程：x^2 y'' + x y' - y = 0
ode8 = sp.Eq(x**2 * f(x).diff(x,2) + x * f(x).diff(x) - f(x), 0)
sol8 = sp.dsolve(ode8)
print("x^2 y'' + x y' - y = 0 的通解：", sol8)

# 二阶变系数齐次：y'' + x y = 0
ode9 = sp.Eq(f(x).diff(x,2) + x*f(x), 0)
sol9 = sp.dsolve(ode9)
print("y'' + x*y = 0 的通解：", sol9)

# 简单的偏微分方程分离变量例子：u_t = u_xx
t = sp.Symbol('t')
u = sp.Function('u')
pde1 = sp.Eq(u(x,t).diff(t), u(x,t).diff(x,2))
print("热传导PDE结构 u_t = u_xx ：", pde1)

# 泰勒展开（ODE级数近似解）
fexpr = sp.Function('f')(x)
ode10 = fexpr.diff(x) + x*fexpr - 1
sol10_series = sp.series(sp.dsolve(ode10), x, 0, 6)
print("级数近似解（ODE）：", sol10_series)

# 三阶常系数齐次：y''' - 3y'' + 3y' - y = 0
ode11 = sp.Eq(f(x).diff(x,3) - 3*f(x).diff(x,2) + 3*f(x).diff(x,1) - f(x), 0)
sol11 = sp.dsolve(ode11)
print("y''' - 3y'' + 3y' - y = 0 的通解：", sol11)
