# -*- coding: utf-8 -*-
"""Handbeispiele zum Kerntrick (Kopie der Prüfungen aus der Planung, `handbeispiele_01.py`): XOR mit vier Punkten, von Hand rechenbar, exakt in Brüchen.
Unabhängig von den Modulen der App (nur Standardbibliothek, numpy, scipy.linprog und scikit-learn als Fremdreferenz); `tests/test_handbeispiele.py` ruft die Teile auf
und vergleicht sie mit den Ergebnissen der App. Konvention: Bias als Zusatzkoordinate 1, im Kern also Kt = K + 1; Marge = 2/||w|| (Breite), Abstand = 1/||w||.
"""
import itertools
import math
import os
from fractions import Fraction as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

N = [0]


def ok(cond, msg=""):
    assert cond, msg
    N[0] += 1


# ---------------------------------------------------------------- Teil 1: Kerntrick von Hand (Polynomkern Grad 2 in 2D, exakt)
def poly_k(x, y, c=1):
    return (x[0] * y[0] + x[1] * y[1] + c) ** 2


def feat_weighted_dot(x, y, c=1):
    """phi(x).phi(y) mit phi = (c, sqrt(2c) x1, sqrt(2c) x2, x1^2, x2^2, sqrt2 x1 x2): die Wurzeln stecken in ganzzahligen Gewichten (c^2... siehe unten),
    so bleibt die Rechnung in Brüchen. Gewichte der Monome (1, x1, x2, x1^2, x2^2, x1 x2) im Skalarprodukt: (c^2, 2c, 2c, 1, 1, 2)."""
    mx = (1, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1])
    my = (1, y[0], y[1], y[0] ** 2, y[1] ** 2, y[0] * y[1])
    w = (c * c, 2 * c, 2 * c, 1, 1, 2)
    return sum(wi * a * b for wi, a, b in zip(w, mx, my))


def teil1():
    pts = [(F(1), F(1)), (F(-1), F(-1)), (F(1), F(-1)), (F(-1), F(1)), (F(1, 2), F(3)), (F(-2, 3), F(5, 7)), (F(0), F(0)), (F(7), F(-3, 4))]
    for x in pts:
        for y in pts:
            for c in (1, 2):
                ok(feat_weighted_dot(x, y, c) == poly_k(x, y, c), "Multinomialsatz")
    ok(math.comb(2 + 2, 2) == 6 and math.comb(10 + 3, 3) == 286 and math.comb(103, 3) == 176851, "Merkmalsdimension C(p+d,d)")
    # Von Hand: x = (1,1), y = (1,-1): x.y = 0 -> Kern 1; (-1,-1) mit (1,1): x.y = -2 -> Kern 1; x mit sich: 2 -> 9
    ok(poly_k((1, 1), (1, -1)) == 1 and poly_k((1, 1), (-1, -1)) == 1 and poly_k((1, 1), (1, 1)) == 9, "Einträge der Gram-Matrix")


# ---------------------------------------------------------------- Teil 2: XOR ist linear nicht trennbar
XOR = [((1, 1), 1), ((-1, -1), 1), ((1, -1), -1), ((-1, 1), -1)]  # Punkte a, b, c, d
NAMES = "abcd"


def teil2():
    # Von Hand: y_i (w.x_i + b) >= 1. a: w1 + w2 + b >= 1; b: -w1 - w2 + b >= 1 (Summe: 2b >= 2, also b >= 1);
    # c: -(w1 - w2 + b) >= 1; d: -(-w1 + w2 + b) >= 1 (Summe: -2b >= 2, also b <= -1): Widerspruch.
    rows = []
    for (x, y) in XOR:
        # y (w1 x1 + w2 x2 + b) >= 1 als Koeffizientenvektor (w1, w2, b)
        rows.append((y * x[0], y * x[1], y))
    s_ab = tuple(rows[0][k] + rows[1][k] for k in range(3))
    s_cd = tuple(rows[2][k] + rows[3][k] for k in range(3))
    ok(s_ab == (0, 0, 2) and s_cd == (0, 0, -2), "Summen der Randbedingungen: 2b >= 2 und -2b >= 2")
    # Fremdweg: Lineares Programm (HiGHS) unzulässig
    from scipy.optimize import linprog
    A = -np.array(rows, float)
    res = linprog(c=np.zeros(3), A_ub=A, b_ub=-np.ones(4), bounds=[(None, None)] * 3, method="highs")
    ok(res.status == 2, "LP unzulässig (HiGHS-Status 2)")
    # Exakter Beweis, dass das lineare Perzeptron nie konvergiert: ganzzahlige Gewichte (w1, w2, b), feste Reihenfolge a, b, c, d;
    # kehrt der Zustand am Epochenanfang wieder, wiederholt sich alles: Zyklus.
    w = (0, 0, 0)
    seen = {}
    ep = 0
    conv = False
    while w not in seen:
        seen[w] = ep
        mist = 0
        for (x, y) in XOR:
            phi = (x[0], x[1], 1)
            if y * sum(wi * pi for wi, pi in zip(w, phi)) <= 0:
                w = tuple(wi + y * pi for wi, pi in zip(w, phi))
                mist += 1
        ep += 1
        if mist == 0:
            conv = True
            break
    ok(not conv, "lineares Perzeptron konvergiert auf XOR nicht")
    ok(w in seen, "Zustand am Epochenanfang kehrt wieder (Zyklus)")
    return ep - seen[w], len(seen)


# ---------------------------------------------------------------- Teil 3: das Kern-Perzeptron auf XOR von Hand
def gram(kfun, pts):
    return [[kfun(p, q) for q in pts] for p in pts]


def dual_perceptron_exact(Kt, y, order, max_epochs=50):
    """Duales Perzeptron in exakten Brüchen. Rückgabe (Aktualisierungen, Epochen, konvergiert, alpha, Protokoll)."""
    n = len(y)
    alpha = [0] * n
    log = []
    ups = 0
    for ep in range(1, max_epochs + 1):
        mist = 0
        for i in order:
            f = sum(alpha[j] * y[j] * Kt[j][i] for j in range(n))
            if y[i] * f <= 0:
                alpha[i] += 1
                ups += 1
                mist += 1
                log.append((ep, NAMES[i], f))
        if mist == 0:
            return ups, ep, True, alpha, log
    return ups, max_epochs, False, alpha, log


def teil3():
    pts = [p for p, _ in XOR]
    y = [yy for _, yy in XOR]
    K = gram(poly_k, pts)
    Kt = [[K[i][j] + 1 for j in range(4)] for i in range(4)]
    ok(all(K[i][j] == (9 if i == j else 1) for i in range(4) for j in range(4)), "K = 8 I + 1 1'")
    ok(all(Kt[i][j] == (10 if i == j else 2) for i in range(4) for j in range(4)), "Kt = 8 I + 2 1 1'")
    # Von Hand in der Reihenfolge a, b, c, d: f_i = 8 alpha_i y_i + 2 S mit S = sum alpha_j y_j
    ups, ep, conv, alpha, log = dual_perceptron_exact(Kt, y, [0, 1, 2, 3])
    ok(conv and ups == 4 and alpha == [1, 1, 1, 1], "vier Aktualisierungen, alpha = (1, 1, 1, 1)")
    ok(ep == 3, "dritte Epoche ohne Fehler, also drei Durchläufe insgesamt")
    ok([(e, n) for e, n, _ in log] == [(1, "a"), (1, "c"), (1, "d"), (2, "b")], "Reihenfolge der Aktualisierungen: a, c, d in Epoche 1, b in Epoche 2")
    ok([f for _, _, f in log] == [0, 2, 0, -2], "Werte f(x_i) bei den Aktualisierungen: 0, 2, 0, -2")
    # f(x) = sum alpha_j y_j (x_j.x + 1)^2 + sum alpha_j y_j = 8 x1 x2 (exakt, an vielen rationalen Punkten)
    pts_test = [(F(a, 3), F(b, 5)) for a in range(-6, 7, 2) for b in range(-10, 11, 4)]
    for x in pts_test:
        f = sum(alpha[j] * y[j] * (poly_k(pts[j], x) + 1) for j in range(4))
        ok(f == 8 * x[0] * x[1], "f(x) = 8 x1 x2")
    # Marge: f(x_i) y_i = 8 an allen vier Punkten
    ok(all(y[i] * sum(alpha[j] * y[j] * Kt[j][i] for j in range(4)) == 8 for i in range(4)), "y_i f(x_i) = 8")
    # Marge der maximalen Trennung: alpha_svm = 1/8 (Kt y = 8 y, also Q alpha = 1), ||w||^2 = sum alpha = 1/2, Abstand 1/||w|| = sqrt 2, Breite 2 sqrt 2
    a8 = [F(1, 8)] * 4
    Qa = [sum(y[i] * y[j] * Kt[i][j] * a8[j] for j in range(4)) for i in range(4)]
    ok(Qa == [1, 1, 1, 1] and sum(a8[j] * y[j] for j in range(4)) == 0, "KKT: Q alpha = 1, Summe alpha y = 0, alpha > 0")
    w2 = sum(a8)  # ||w||^2 = alpha'Q alpha = alpha'1
    ok(w2 == F(1, 2), "||w||^2 = 1/2")
    R2 = max(Kt[i][i] for i in range(4))
    bound = R2 * w2  # (R/gamma)^2 mit gamma = 1/||w||
    ok(R2 == 10 and bound == 5, "R^2 = 10, Novikoff-Schranke R^2 ||w||^2 = 5")
    ok(ups <= bound, "4 Aktualisierungen <= 5")
    # Fremdreferenz SVC (Gram vorgegeben, großes C)
    from sklearn.svm import SVC
    Ktf = np.array(Kt, float)
    m = SVC(kernel="precomputed", C=1e4, tol=1e-10).fit(Ktf, np.array(y))
    al = np.zeros(4)
    al[m.support_] = np.abs(m.dual_coef_[0])
    ok(np.allclose(al, 0.125, atol=1e-6), "SVC: alpha = 1/8")
    # Alle 24 Reihenfolgen (feste Reihenfolge je Epoche): Aktualisierungen, alle <= 5
    counts = []
    for order in itertools.permutations(range(4)):
        u, e, c, a, _ = dual_perceptron_exact(Kt, y, list(order))
        ok(c, "konvergiert")
        ok(u == 4 and a == [1, 1, 1, 1], "vier Korrekturen und dasselbe alpha = (1, 1, 1, 1) in jeder der 24 Reihenfolgen")
        ok(u <= bound, "Schranke gilt für jede Reihenfolge")
        counts.append(u)
    # Novikoff-Beweisschritte, nur mit der Gram-Matrix: nach k Aktualisierungen ||w||^2 <= k R^2 und w.w_hat >= k gamma
    # (w_hat = Richtung der maximalen Trennung; w.w_hat = sum alpha_i y_i f_hat(x_i)/||w_hat||, f_hat = sum a8_j y_j Kt_j.)
    alpha_t = [0] * 4
    k = 0
    gamma2 = 1 / w2  # gamma^2 = 2 (Abstand zum Quadrat); w.w_hat >= k gamma <=> (w.w*)^2 >= k^2 gamma^2 ||w*||^2 ... siehe unten
    for ep_ in range(5):
        for i in range(4):
            f = sum(alpha_t[j] * y[j] * Kt[j][i] for j in range(4))
            if y[i] * f <= 0:
                alpha_t[i] += 1
                k += 1
                nw2 = sum(alpha_t[p] * y[p] * Kt[p][q] * alpha_t[q] * y[q] for p in range(4) for q in range(4))
                ok(nw2 <= k * R2, "||w_k||^2 <= k R^2")
                wstar = sum(alpha_t[p] * y[p] * sum(a8[q] * y[q] * Kt[q][p] for q in range(4)) for p in range(4))  # w_k . w* mit ||w*||^2 = 1/2
                ok(wstar >= k, "w_k . w* >= k (da y_i f*(x_i) = 1 an allen Punkten)")
                ok(wstar * wstar <= nw2 * w2, "Cauchy-Schwarz (w.w*)^2 <= ||w||^2 ||w*||^2")
    ok(k == 4, "vier Aktualisierungen im Beweislauf")
    # Aus k^2 <= (w.w*)^2 <= ||w||^2 ||w*||^2 <= k R^2 ||w*||^2 folgt k <= R^2 ||w*||^2 = 5
    return min(counts), max(counts)


# ---------------------------------------------------------------- Teil 4: Gauß-Kern auf XOR, Marge 1 - exp(-2/sigma^2)
def solve_exact(A, b):
    n = len(A)
    M = [list(map(F, row)) + [F(bb)] for row, bb in zip(A, b)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                fct = M[r][c]
                M[r] = [vr - fct * vc for vr, vc in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def teil4():
    y = [yy for _, yy in XOR]
    # Gauß-Kern exp(-r^2/(2 sigma^2)); XOR-Abstände: a-b und c-d: r^2 = 8, sonst r^2 = 4. Mit u = exp(-2/sigma^2): K_ac = u, K_ab = u^2.
    # Für jedes rationale u in (0,1) rechnet das KKT-System exakt: Kein Fremdlöser.
    for u in (F(1, 5), F(1, 2), F(9, 10)):
        K = [[1 if i == j else (u * u if {i, j} in ({0, 1}, {2, 3}) else u) for j in range(4)] for i in range(4)]
        # KKT: sum_j Q_ij alpha_j + y_i b = 1 (alle vier Stützvektoren), sum_j y_j alpha_j = 0
        Q = [[y[i] * y[j] * K[i][j] for j in range(4)] for i in range(4)]
        A = [Q[i] + [y[i]] for i in range(4)] + [[y[j] for j in range(4)] + [0]]
        sol = solve_exact(A, [1, 1, 1, 1, 0])
        al, b = sol[:4], sol[4]
        ok(al == [1 / (1 - u) ** 2] * 4 and b == 0, "alpha = 1/(1-u)^2, b = 0")
        w2 = sum(al[i] * al[j] * y[i] * y[j] * K[i][j] for i in range(4) for j in range(4))
        ok(w2 == 4 / (1 - u) ** 2, "||w||^2 = 4/(1-u)^2")
        # Marge 2/||w|| = 1 - u (Quadrat: 4/||w||^2 = (1-u)^2)
        ok(4 / w2 == (1 - u) ** 2, "Marge^2 = (1-u)^2")
        ok(all(a > 0 for a in al), "alle vier Punkte sind Stützvektoren")
    # Zahlenwerte für sigma = 1, 2 (gerundet) gegen SVC (Fremdreferenz) und gegen die Eigenzerlegung der Gram-Matrix (zweiter Weg)
    from sklearn.svm import SVC
    for sigma, exp_marge in ((1.0, 0.864665), (2.0, 0.393469)):
        u = math.exp(-2 / sigma ** 2)
        ok(abs((1 - u) - exp_marge) < 1e-6, "gedruckte Marge aus V1")
        pts = np.array([p for p, _ in XOR], float)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        K = np.exp(-d2 / (2 * sigma ** 2))
        yy = np.array(y, float)
        m = SVC(kernel="precomputed", C=1e6, tol=1e-10).fit(K, yy)
        al = np.zeros(4)
        al[m.support_] = np.abs(m.dual_coef_[0])
        w2 = float((al * yy) @ K @ (al * yy))
        ok(abs(2 / math.sqrt(w2) - (1 - u)) < 1e-6, "SVC-Marge = 1 - exp(-2/sigma^2)")
        # Eigenzerlegung: y ist Eigenvektor von K zum Eigenwert (1-u)^2; daraus alpha = 1/(1-u)^2
        ev, V = np.linalg.eigh(K)
        ok(abs(yy @ K @ yy / (yy @ yy) - (1 - u) ** 2) < 1e-12, "Rayleigh-Quotient von y = (1-u)^2")
        ok(min(abs(ev - (1 - u) ** 2)) < 1e-12, "(1-u)^2 ist Eigenwert")
    # Kern-Perzeptron mit Gauß-Kern (Bias als +1) in der Reihenfolge a, b, c, d, in Gleitkomma; Schranke R^2 ||w||^2 mit R^2 = 2
    out = {}
    for sigma in (0.5, 1.0, 2.0, 4.0):
        u = math.exp(-2 / sigma ** 2)
        pts = np.array([p for p, _ in XOR], float)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        Kt = np.exp(-d2 / (2 * sigma ** 2)) + 1.0
        yy = np.array(y, float)
        alpha = np.zeros(4)
        ups = 0
        for ep in range(1, 1000):
            mist = 0
            for i in range(4):
                if yy[i] * ((alpha * yy) @ Kt[:, i]) <= 0:
                    alpha[i] += 1
                    ups += 1
                    mist += 1
            if mist == 0:
                break
        # ||w||^2 der maximalen Trennung = Summe der alpha = 4/(1-u)^2 (Teil 4, exakt); Kt = K + 1 ändert die Lösung nicht (Summe alpha y = 0)
        w2 = 4 / (1 - u) ** 2
        bound = 2.0 * w2  # R^2 = max Kt_ii = 2
        ok(ups <= bound + 1e-9, "Schranke für den Gauß-Kern auf XOR")
        out[sigma] = (ups, bound)
    return out


# ---------------------------------------------------------------- Teil 3b: zwei Maximalmargen, Bias als Koordinate gegen freies b (exakt)
def teil3b():
    """Zwei Punkte x = 0 (y = -1) und x = 1 (y = +1), linearer Kern. Marge mit freiem b (Stück 2) und Marge in (phi, 1) (Bias als Koordinate, Kt = K + 1) sind verschieden."""
    y = [-1, 1]
    pts = [F(0), F(1)]
    K = [[a * b for b in pts] for a in pts]
    # freies b: KKT beider Punkte als Stützvektoren: Summe_j Q_ij alpha_j + y_i b = 1, Summe y alpha = 0 (3 Unbekannte alpha0, alpha1, b)
    Q = [[y[i] * y[j] * K[i][j] for j in range(2)] for i in range(2)]
    # Zeile 0: 0*a0 + 0*a1 - b = 1 -> b = -1; Zeile 1: 0*a0 + 1*a1 + b = 1 -> a1 = 2; a0 = a1 (Summe y alpha = 0)
    a1 = F(2)
    a0 = a1
    b = F(-1)
    ok(all(sum(Q[i][j] * (a0, a1)[j] for j in range(2)) + y[i] * b == 1 for i in range(2)) and y[0] * a0 + y[1] * a1 == 0, "KKT mit freiem b erfüllt: alpha = (2, 2), b = -1")
    w2_frei = sum((a0, a1)[i] * (a0, a1)[j] * Q[i][j] for i in range(2) for j in range(2))
    ok(w2_frei == 4, "||w||^2 = 4 mit freiem b, also Marge 2/||w|| = 1 = Abstand der beiden Punkte")
    # Bias als Koordinate: Kt = K + 1, keine Gleichung Summe y alpha = 0 (Q~ alpha = 1 für beide Stützvektoren)
    Qt = [[y[i] * y[j] * (K[i][j] + 1) for j in range(2)] for i in range(2)]
    det = Qt[0][0] * Qt[1][1] - Qt[0][1] * Qt[1][0]
    ok(det == 1, "det Q~ = 1")
    al = [(Qt[1][1] - Qt[0][1]) / det, (Qt[0][0] - Qt[1][0]) / det]  # Q~^{-1} (1, 1)
    ok(al == [3, 2] and all(a > 0 for a in al), "alpha = (3, 2), beide positiv: beide Punkte sind Stützvektoren")
    w2_tilde = sum(al[i] * al[j] * Qt[i][j] for i in range(2) for j in range(2))
    ok(w2_tilde == 5 == sum(al), "||w~||^2 = 5 in (phi, 1) (= Summe alpha)")
    ok(4 / w2_tilde == F(4, 5) and 4 / w2_frei == 1, "Marge^2: 4/5 in (phi, 1) gegen 1 mit freiem b: Verhältnis sqrt(4/5) = 0,894, nicht 1")
    # Gegenweg: Minimierung von w^2 + b^2 (Bias in der Norm) über das zulässige Gebiet b <= -1, w >= 1 - b, auf einem feinen rationalen Gitter in b
    best = min((1 - bb) ** 2 + bb ** 2 for bb in [F(-k, 100) for k in range(100, 400)])
    ok(best == 5, "Minimum von (1 - b)^2 + b^2 über b <= -1 ist 5 (bei b = -1)")
    # Gleichheit bei XOR: Summe alpha y = 0, daher beide Maxima gleich (Teil 3: ||w||^2 = 1/2 mit und ohne Bias)


def main():
    teil1()
    zyk, nz = teil2()
    lo, hi = teil3()
    teil3b()
    g = teil4()
    print(f"Teil 2: lineares Perzeptron auf XOR: Zyklus der Länge {zyk} Epochen nach {nz} verschiedenen Zuständen am Epochenanfang")
    print(f"Teil 3: XOR, Polynomkern Grad 2, Reihenfolge a,b,c,d: 4 Aktualisierungen, alpha = (1,1,1,1), f(x) = 8 x1 x2, Schranke 5; alle 24 Reihenfolgen: {lo} bis {hi} Aktualisierungen")
    for s, (u, b) in g.items():
        print(f"Teil 4: Gauß sigma = {s}: {u} Aktualisierungen, Schranke R^2 ||w||^2 = {b:.2f}")
    print(f"ALLE PRÜFUNGEN BESTANDEN ({N[0]} Prüfungen)")


if __name__ == "__main__":
    main()
