"""
BIG-M METHOD (Penalty Method) - Linear Programming Solver
------------------------------------------------------------
Problem (a well-known Big-M textbook example):

    Minimize   Z = 4x1 + x2

    Subject to:
        3x1 +  x2  = 3        (equality constraint  -> needs artificial var)
        4x1 + 3x2 >= 6        (>= constraint        -> needs surplus + artificial var)
         x1 + 2x2 <= 4        (<= constraint        -> needs slack var)
        x1, x2 >= 0

Standard form after adding slack (S), surplus (S), and artificial (A) variables:

    3x1 +  x2 + A1                = 3
    4x1 + 3x2      - S1      + A2 = 6
     x1 + 2x2            + S2     = 4

Big-M objective (minimization):
    Z = 4x1 + x2 + 0.S1 + 0.S2 + M.A1 + M.A2
"""

from fractions import Fraction as F

M = 1_000_000  # a sufficiently large penalty (Big-M), used only for ranking, kept symbolic via large int

# Variable order in the tableau:
#   x1  x2  S1  S2  A1  A2  |  RHS
var_names = ["x1", "x2", "S1", "S2", "A1", "A2"]

# Cost (objective) coefficients for MINIMIZATION
c = [4, 1, 0, 0, M, M]

# Initial tableau rows: [x1, x2, S1, S2, A1, A2, RHS]
# Row 1 : 3x1 +  x2         + A1        = 3
# Row 2 : 4x1 + 3x2  - S1        + A2   = 6
# Row 3 :  x1 + 2x2       + S2          = 4
tableau = [
    [3, 1, 0, 0, 1, 0, 3],
    [4, 3, -1, 0, 0, 1, 6],
    [1, 2, 0, 1, 0, 0, 4],
]

# Basic variables initially in each row (indices into var_names)
basis = [4, 5, 3]   # A1, A2, S2


def print_tableau(tableau, basis, c, iteration):
    print(f"\n{'=' * 90}")
    print(f"ITERATION {iteration}")
    print(f"{'=' * 90}")
    header = f"{'Basis':>8}" + "".join(f"{v:>12}" for v in var_names) + f"{'RHS':>12}"
    print(header)
    for i, row in enumerate(tableau):
        row_str = f"{var_names[basis[i]]:>8}" + "".join(f"{fmt(val):>12}" for val in row[:-1]) + f"{fmt(row[-1]):>12}"
        print(row_str)

    # Compute Zj row and Zj - Cj row
    zj = []
    for j in range(len(var_names)):
        s = sum(c[basis[i]] * tableau[i][j] for i in range(len(tableau)))
        zj.append(s)
    zj_rhs = sum(c[basis[i]] * tableau[i][-1] for i in range(len(tableau)))

    cj_row = "Cj      " + "".join(f"{fmt(c[j]):>12}" for j in range(len(var_names)))
    print(cj_row)

    zj_row = "Zj      " + "".join(f"{fmt(zj[j]):>12}" for j in range(len(var_names))) + f"{fmt(zj_rhs):>12}"
    print(zj_row)

    zc_row_vals = [zj[j] - c[j] for j in range(len(var_names))]
    zc_row = "Zj-Cj   " + "".join(f"{fmt(v):>12}" for v in zc_row_vals)
    print(zc_row)

    return zj, zc_row_vals


def fmt(x):
    """Nicely format Fractions / big-M expressions for display."""
    if isinstance(x, F):
        if x.denominator == 1:
            return str(x.numerator)
        return f"{float(x):.3f}"
    return str(x)


def to_fraction_tableau(tab):
    return [[F(val) for val in row] for row in tab]


def big_m_simplex(tableau, basis, c):
    tableau = to_fraction_tableau(tableau)
    iteration = 0
    while True:
        zj, zc = print_tableau(tableau, basis, c, iteration)

        # Optimality check (minimization): stop when all Zj - Cj <= 0
        entering = None
        best = 0
        for j in range(len(var_names)):
            if zc[j] > best:
                best = zc[j]
                entering = j

        if entering is None:
            print("\nOptimality reached: all (Zj - Cj) <= 0.")
            break

        print(f"\nEntering variable : {var_names[entering]}  (largest positive Zj-Cj = {fmt(best)})")

        # Ratio test (minimum ratio, only positive entries in entering column)
        ratios = []
        for i in range(len(tableau)):
            col_val = tableau[i][entering]
            if col_val > 0:
                ratios.append((tableau[i][-1] / col_val, i))
            else:
                ratios.append((None, i))

        valid_ratios = [(r, i) for r, i in ratios if r is not None]
        if not valid_ratios:
            print("Problem is UNBOUNDED.")
            return None, None

        min_ratio, leaving_row = min(valid_ratios, key=lambda t: t[0])
        print(f"Leaving variable  : {var_names[basis[leaving_row]]}  (min ratio = {fmt(min_ratio)})")

        # Pivot
        pivot_val = tableau[leaving_row][entering]
        tableau[leaving_row] = [val / pivot_val for val in tableau[leaving_row]]
        for i in range(len(tableau)):
            if i != leaving_row:
                factor = tableau[i][entering]
                if factor != 0:
                    tableau[i] = [tableau[i][j] - factor * tableau[leaving_row][j] for j in range(len(tableau[i]))]

        basis[leaving_row] = entering
        iteration += 1

    return tableau, basis


if __name__ == "__main__":
    print("BIG-M SIMPLEX METHOD")
    print("Minimize Z = 4x1 + x2")
    print("s.t.  3x1 +  x2      = 3")
    print("      4x1 + 3x2     >= 6")
    print("       x1 + 2x2     <= 4")
    print("      x1, x2 >= 0")

    final_tableau, final_basis = big_m_simplex(tableau, basis, c)

    if final_tableau is not None:
        print(f"\n{'=' * 90}")
        print("OPTIMAL SOLUTION")
        print(f"{'=' * 90}")
        solution = {v: 0 for v in var_names}
        for i, b in enumerate(final_basis):
            solution[var_names[b]] = final_tableau[i][-1]

        for v in ["x1", "x2"]:
            print(f"  {v} = {fmt(solution[v])}")

        z_value = 4 * solution["x1"] + 1 * solution["x2"]
        print(f"\n  Optimal Objective Value  Z = 4x1 + x2 = {fmt(z_value)}")

        # sanity check: no artificial variable should remain in the basis with positive value
        art_in_basis = any(var_names[b] in ("A1", "A2") and final_tableau[i][-1] != 0
                            for i, b in enumerate(final_basis))
        if art_in_basis:
            print("\n  WARNING: an artificial variable remained positive in the basis -> INFEASIBLE.")
        else:
            print("\n  Feasibility check passed (no artificial variable remains positive).")
