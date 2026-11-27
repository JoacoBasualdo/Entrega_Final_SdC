"""
Ejercicio 1 - Respuesta al escalón a lazo abierto
G(s)H(s) = 10 e^{-20 s} / (1 + 0.0125 s)
El retardo se aproxima por Padé (orden 1, 2 y 3) y se compara con la
respuesta exacta (retardo puro desplazando la respuesta de primer orden).
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# ---------------- Parámetros de la planta ----------------
K, tau, L = 10.0, 0.0125, 20.0
G0 = ct.tf([K], [tau, 1])               # parte sin retardo

t = np.linspace(0, 120, 400001)          # paso fino: tau es muy chico


def metricas(t, y, yf, tol=0.02):
    """Parámetros de la respuesta al escalón."""
    i_min, i_max = np.argmin(y), np.argmax(y)
    sub = max(0.0, -y[i_min]) / yf * 100           # subpico (respuesta inversa)
    mp = max(0.0, y[i_max] - yf) / yf * 100        # sobrepico
    # Tiempo de subida 10-90 % medido sobre el ascenso definitivo
    # (después del subpico, para no contar el salto inicial de Padé)
    i90 = i_min + np.argmax(y[i_min:] >= 0.9 * yf)
    i10 = np.where(y[:i90] < 0.1 * yf)[0][-1] + 1
    i50 = np.where(y[:i90] < 0.5 * yf)[0][-1] + 1
    t10, t90, td = t[i10], t[i90], t[i50]
    fuera = np.where(np.abs(y - yf) > tol * yf)[0]
    ts = t[fuera[-1] + 1] if fuera.size else 0.0
    return dict(yf=yf, ymin=y[i_min], tmin=t[i_min], sub=sub, Mp=mp,
                tp=t[i_max] if mp > 0 else np.nan,
                t10=t10, t90=t90, tr=t90 - t10, td=td, ts=ts)


resultados = {}

# ---------------- Respuesta exacta ----------------
y_ex = K * (1 - np.exp(-np.clip(t - L, 0, None) / tau))
resultados["Exacta"] = (y_ex, metricas(t, y_ex, K))

# ---------------- Aproximaciones de Padé ----------------
for n in (1, 2, 3):
    num, den = ct.pade(L, n)
    G = G0 * ct.tf(num, den)
    _, y = ct.step_response(G, T=t)
    resultados[f"Padé {n}"] = (y, metricas(t, y, ct.dcgain(G)))
    if n == 1:
        print("G(s)H(s) con Padé 1:", G)
        print("Polos:", ct.poles(G), " Ceros:", ct.zeros(G))

# ---------------- Tabla de resultados ----------------
print(f"{'Modelo':10s} {'y_final':>8s} {'y_min':>8s} {'t_min':>7s} "
      f"{'Subpico%':>9s} {'Mp%':>6s} {'tp':>7s} {'td':>7s} {'tr':>7s} {'ts(2%)':>7s}")
for nombre, (_, m) in resultados.items():
    print(f"{nombre:10s} {m['yf']:8.3f} {m['ymin']:8.3f} {m['tmin']:7.3f} "
          f"{m['sub']:9.2f} {m['Mp']:6.2f} {m['tp']:7.3f} {m['td']:7.3f} {m['tr']:7.3f} {m['ts']:7.3f}")

# ---------------- Gráficos ----------------
fig, ax = plt.subplots(figsize=(8, 4.5))
estilos = {"Exacta": "k-", "Padé 1": "C0--", "Padé 2": "C1-.", "Padé 3": "C2:"}
for nombre, (y, _) in resultados.items():
    ax.plot(t, y, estilos[nombre], lw=1.6, label=nombre)
ax.axhline(K, color="gray", lw=0.8, ls="--")
ax.set(xlabel="Tiempo [s]", ylabel="Salida y(t)", xlim=(0, 80),
       title="Respuesta al escalón unitario a lazo abierto")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()

# Detalle del arranque (respuesta inversa)
fig, ax = plt.subplots(figsize=(8, 4.5))
for nombre, (y, _) in resultados.items():
    ax.plot(t, y, estilos[nombre], lw=1.6, label=nombre)
ax.set(xlabel="Tiempo [s]", ylabel="Salida y(t)", xlim=(0, 0.2),
       title="Detalle de los primeros instantes (respuesta inversa)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()

plt.show()