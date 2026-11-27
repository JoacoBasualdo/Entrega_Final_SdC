
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path

# Ejercicio 8 - Realimentacion de estados

# Matrices del sistema obtenidas en el ejercicio 6
A = np.array([[0, 1],
              [-8, -80.1]])

B = np.array([[0],
              [1]])

C = np.array([[80, -800]])

D = np.array([[0]])

# Especificaciones de diseno
Mp = 0.10
Ts = 120

# Calculo del coeficiente de amortiguamiento
zeta = -np.log(Mp) / np.sqrt(np.pi**2 + np.log(Mp)**2)

# Calculo de la frecuencia natural
wn = 4 / (zeta * Ts)

# Polos deseados
p1 = -zeta * wn + 1j * wn * np.sqrt(1 - zeta**2)
p2 = -zeta * wn - 1j * wn * np.sqrt(1 - zeta**2)

# Calculo de las ganancias de realimentacion
k1 = wn**2 - 8
k2 = 2 * zeta * wn - 80.1

K = np.array([[k1, k2]])

# Sistema compensado
Acl = A - B @ K

sys_cl = ct.ss(Acl, B, C, D)

# Funcion de transferencia a lazo cerrado
T_cl = ct.ss2tf(sys_cl)

# Simulacion de la respuesta al escalon unitario
t = np.linspace(0, 300, 30001)

t, y = ct.step_response(sys_cl, T=t)

# Valor final
yss = float(ct.dcgain(sys_cl))

# Parametros de la respuesta temporal
info = ct.step_info(
    sys_cl,
    T=t,
    SettlingTimeThreshold=0.02
)

# Resultados
np.set_printoptions(precision=10)

print("Coeficiente de amortiguamiento:", zeta)
print("Frecuencia natural:", wn)

print("\nPolos deseados:")
print(p1)
print(p2)

print("\nMatriz de realimentacion K:")
print(K)

print("\nMatriz A a lazo cerrado:")
print(Acl)

print("\nPolos del sistema compensado:")
print(np.linalg.eigvals(Acl))

print("\nFuncion de transferencia a lazo cerrado:")
print(T_cl)

print("\nParametros de la respuesta temporal:")
print(f"Sobrepico: {info['Overshoot']:.2f} %")
print(f"Tiempo de pico: {info['PeakTime']:.2f} s")
print(f"Tiempo de establecimiento: {info['SettlingTime']:.2f} s")
print(f"Valor final: {yss:.4f}")

# Respuesta normalizada respecto de su valor final
yn = y / yss

# Graficos
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Respuesta temporal real
ax[0].plot(t, y, color="tab:blue",
           label="Sistema compensado")

ax[0].axhline(yss, color="gray", linestyle="--",
              label="Valor final")

ax[0].axhline(1, color="red", linestyle=":",
              label="Referencia unitaria")

ax[0].set_title("Respuesta al escalon unitario")
ax[0].set_xlabel("Tiempo [s]")
ax[0].set_ylabel("Salida y(t)")
ax[0].grid(True)
ax[0].legend()

# Respuesta temporal normalizada
ax[1].plot(t, yn, color="tab:blue",
           label="Respuesta normalizada")

ax[1].axhline(1, color="gray", linestyle="--",
              label="Valor final normalizado")

ax[1].axhspan(0.98, 1.02, color="green", alpha=0.15,
              label="Banda del 2%")

ax[1].axhline(1.10, color="purple", linestyle=":",
              label="Sobrepico deseado")

ax[1].axvline(info["SettlingTime"],
              color="orange", linestyle="--",
              label="Tiempo de establecimiento obtenido")

ax[1].axvline(120, color="red", linestyle=":",
              label="Tiempo de establecimiento deseado")

ax[1].set_title("Respuesta al escalon normalizada")
ax[1].set_xlabel("Tiempo [s]")
ax[1].set_ylabel("Salida normalizada")
ax[1].grid(True)
ax[1].legend(loc="lower right")

plt.tight_layout()

# Guardar imagen para el informe
Path("Imagenes").mkdir(exist_ok=True)

plt.savefig("Imagenes/Ej8.png",
            dpi=300, bbox_inches="tight")

plt.show()
