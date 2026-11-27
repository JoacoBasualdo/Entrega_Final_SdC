
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path

# Ejercicio 10 - Expansion integral

# Matrices del sistema original
A = np.array([[0, 1],
              [-8, -80.1]])

B = np.array([[0],
              [1]])

C = np.array([[80, -800]])

D = np.array([[0]])

# =====================================================
# 1. SISTEMA EXPANDIDO
# =====================================================

Ae = np.array([[0, 1, 0],
               [-8, -80.1, 0],
               [-80, 800, 0]])

Be = np.array([[0],
               [1],
               [0]])

Br = np.array([[0],
               [0],
               [1]])

Ce = np.array([[80, -800, 0]])

# Verificacion de la controlabilidad
Mc = ct.ctrb(Ae, Be)

print("Matriz de controlabilidad:")
print(Mc)

print("Determinante:", np.linalg.det(Mc))
print("Rango:", np.linalg.matrix_rank(Mc))

# =====================================================
# 2. ESPECIFICACIONES DE DISENO
# =====================================================

Mp = 0.10
Ts = 120

# Coeficiente de amortiguamiento
zeta = -np.log(Mp) / np.sqrt(
    np.pi**2 + np.log(Mp)**2
)

# Frecuencia natural
wn = 4 / (zeta * Ts)

# Polos dominantes del ejercicio 8
p1 = -zeta * wn + 1j * wn * np.sqrt(1 - zeta**2)
p2 = -zeta * wn - 1j * wn * np.sqrt(1 - zeta**2)

# Tercer polo
p3 = -5 * zeta * wn

# =====================================================
# 3. CALCULO DE LAS GANANCIAS
# =====================================================

# Polinomio caracteristico deseado:
# Pd(s) = s^3 + a2*s^2 + a1*s + a0

a2 = 2 * zeta * wn - p3

a1 = wn**2 - 2 * zeta * wn * p3

a0 = -p3 * wn**2

# Ganancias por comparacion de coeficientes
ki = a0 / 80

k1 = a1 + 800 * ki - 8

k2 = a2 - 80.1

# Matriz de realimentacion expandida
Ke = np.array([[k1, k2, -ki]])

# =====================================================
# 4. SISTEMA COMPENSADO
# =====================================================

Acl = Ae - Be @ Ke

# La entrada del sistema compensado es la referencia
sys_cl = ct.ss(Acl, Br, Ce, D)

# Funcion de transferencia
T_cl = ct.ss2tf(sys_cl)

# =====================================================
# 5. SIMULACION
# =====================================================

t = np.linspace(0, 350, 35001)

# Respuesta al escalon unitario
t, y = ct.step_response(sys_cl, T=t)

# Referencia unitaria
r = np.ones_like(t)

# Error temporal
e = r - y

# Valor final
yss = float(ct.dcgain(sys_cl))

# Error de estado estacionario
ess = 1 - yss

# Parametros temporales
info = ct.step_info(
    sys_cl,
    T=t,
    SettlingTimeThreshold=0.02
)

# =====================================================
# 6. RESULTADOS
# =====================================================

np.set_printoptions(precision=10)

print("\nPolos deseados:")
print(p1)
print(p2)
print(p3)

print("\nGanancias de realimentacion:")
print(f"k1 = {k1:.10f}")
print(f"k2 = {k2:.10f}")
print(f"ki = {ki:.10e}")

print("\nMatriz A a lazo cerrado:")
print(Acl)

print("\nPolos obtenidos:")
print(np.linalg.eigvals(Acl))

print("\nFuncion de transferencia:")
print(T_cl)

print("\nParametros temporales:")

print(f"Sobrepico: {info['Overshoot']:.2f} %")

print(f"Tiempo de pico: {info['PeakTime']:.2f} s")

print(
    f"Tiempo de establecimiento: "
    f"{info['SettlingTime']:.2f} s"
)

print(f"Valor final: {yss:.6f}")

print(f"Error estacionario: {ess:.6e}")

# =====================================================
# 7. GRAFICOS
# =====================================================

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Respuesta temporal
ax[0].plot(
    t, y,
    color="tab:blue",
    linewidth=2,
    label="Sistema con expansion integral"
)

ax[0].axhline(
    1,
    color="gray",
    linestyle="--",
    label="Referencia unitaria"
)

# Banda de establecimiento del 2%
ax[0].axhspan(
    0.98, 1.02,
    color="green",
    alpha=0.15,
    label="Banda del 2%"
)

# Sobrepico deseado
ax[0].axhline(
    1.10,
    color="purple",
    linestyle=":",
    label="Sobrepico deseado"
)

# Tiempo de establecimiento obtenido
ax[0].axvline(
    info["SettlingTime"],
    color="orange",
    linestyle="--",
    label="Tiempo de establecimiento"
)

# Punto de maximo sobrepico
ax[0].scatter(
    info["PeakTime"],
    np.max(y),
    color="tab:blue",
    zorder=5
)

ax[0].set_title(
    "Respuesta al escalon con expansion integral"
)

ax[0].set_xlabel("Tiempo [s]")
ax[0].set_ylabel("Salida y(t)")
ax[0].set_xlim(0, 250)
ax[0].grid(True)
ax[0].legend(loc="lower right")

# Evolucion temporal del error
ax[1].plot(
    t, e,
    color="tab:red",
    linewidth=2,
    label="Error e(t)"
)

ax[1].axhline(
    0,
    color="gray",
    linestyle="--",
    label="Error nulo"
)

ax[1].axhspan(
    -0.02, 0.02,
    color="green",
    alpha=0.15,
    label="Banda del 2%"
)

ax[1].set_title("Evolucion temporal del error")
ax[1].set_xlabel("Tiempo [s]")
ax[1].set_ylabel("Error e(t)")
ax[1].set_xlim(0, 250)
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()

# Guardar figura para el informe LaTeX
Path("Imagenes").mkdir(exist_ok=True)

plt.savefig(
    "Imagenes/Ej10.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
