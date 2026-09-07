"""
Transportation Problem: Vogel's Approximation Method (VAM) + MODI Method
--------------------------------------------------------------------------
Case study: A company has 3 factories (sources) supplying goods to
4 warehouses (destinations).

Supply (Factories) : S1 = 50, S2 = 60, S3 = 50   -> total = 160
Demand (Warehouses): D1 = 30, D2 = 40, D3 = 50, D4 = 40 -> total = 160
(Balanced transportation problem, since total supply = total demand)

Unit transportation costs (Cost[i][j] = cost to ship 1 unit from
source i to destination j):

           D1   D2   D3   D4
    S1  [   4,   6,   8,   8 ]
    S2  [   6,   8,   6,   7 ]
    S3  [   5,   7,   6,   8 ]

Step 1: Use VAM to get an Initial Basic Feasible Solution (IBFS)
Step 2: Use MODI to test optimality and iteratively improve the
        allocation until the optimal (minimum-cost) solution is found.
"""

import copy

# ---------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------
supply_orig = [50, 60, 50]
demand_orig = [30, 40, 50, 40]
cost = [
    [4, 6, 8, 8],
    [6, 8, 6, 7],
    [5, 7, 6, 8],
]
source_names = ["S1", "S2", "S3"]
dest_names = ["D1", "D2", "D3", "D4"]


def print_table(allocation, cost, title):
    rows, cols = len(cost), len(cost[0])
    print(f"\n{title}")
    header = "        " + "".join(f"{d:>10}" for d in dest_names)
    print(header)
    for i in range(rows):
        row_str = f"{source_names[i]:<8}"
        for j in range(cols):
            if allocation[i][j] > 0:
                row_str += f"{str(allocation[i][j])+'('+str(cost[i][j])+')':>10}"
            else:
                row_str += f"{'-('+str(cost[i][j])+')':>10}"
        print(row_str)


def total_cost(allocation, cost):
    return sum(allocation[i][j] * cost[i][j]
               for i in range(len(cost)) for j in range(len(cost[0])))


# ---------------------------------------------------------------------
# STEP 1: Vogel's Approximation Method (VAM)
# ---------------------------------------------------------------------
def vogel_approximation_method(supply, demand, cost):
    supply = supply.copy()
    demand = demand.copy()
    rows, cols = len(supply), len(demand)
    allocation = [[0] * cols for _ in range(rows)]
    row_done = [False] * rows
    col_done = [False] * cols

    step = 1
    while sum(supply) > 0 and sum(demand) > 0:
        row_penalty = [-1] * rows
        col_penalty = [-1] * cols

        for i in range(rows):
            if row_done[i]:
                continue
            costs_row = [cost[i][j] for j in range(cols) if not col_done[j]]
            if len(costs_row) >= 2:
                s = sorted(costs_row)
                row_penalty[i] = s[1] - s[0]
            elif len(costs_row) == 1:
                row_penalty[i] = costs_row[0]

        for j in range(cols):
            if col_done[j]:
                continue
            costs_col = [cost[i][j] for i in range(rows) if not row_done[i]]
            if len(costs_col) >= 2:
                s = sorted(costs_col)
                col_penalty[j] = s[1] - s[0]
            elif len(costs_col) == 1:
                col_penalty[j] = costs_col[0]

        max_row_pen = max(row_penalty)
        max_col_pen = max(col_penalty)

        if max_row_pen >= max_col_pen:
            i = row_penalty.index(max_row_pen)
            j = min((j for j in range(cols) if not col_done[j]),
                    key=lambda j: cost[i][j])
        else:
            j = col_penalty.index(max_col_pen)
            i = min((i for i in range(rows) if not row_done[i]),
                    key=lambda i: cost[i][j])

        qty = min(supply[i], demand[j])
        allocation[i][j] = qty
        print(f"Step {step}: Allocate {qty} units to "
              f"({source_names[i]} -> {dest_names[j]}), cost/unit = {cost[i][j]}")
        step += 1

        supply[i] -= qty
        demand[j] -= qty
        if supply[i] == 0:
            row_done[i] = True
        if demand[j] == 0:
            col_done[j] = True

    return allocation


# ---------------------------------------------------------------------
# STEP 2: MODI (Modified Distribution) Method
# ---------------------------------------------------------------------
def find_closed_loop(start, basic_cells):
    """Find a closed loop (alternating row/column moves) starting and
    ending at `start`, passing only through cells in basic_cells."""
    all_cells = list(basic_cells)
    if start not in all_cells:
        all_cells.append(start)

    def search(path, horizontal):
        last = path[-1]
        candidates = [c for c in all_cells if c != last and
                      ((c[0] == last[0]) if horizontal else (c[1] == last[1]))]
        for nxt in candidates:
            if nxt == start and len(path) >= 3:
                return path + [nxt]
            if nxt in path:
                continue
            result = search(path + [nxt], not horizontal)
            if result:
                return result
        return None

    return search([start], True)


def modi_method(supply, demand, cost, allocation):
    rows, cols = len(supply), len(demand)
    allocation = copy.deepcopy(allocation)
    iteration = 1

    while True:
        basic_cells = [(i, j) for i in range(rows) for j in range(cols)
                       if allocation[i][j] > 0]
        required = rows + cols - 1

        # Handle degeneracy: add a zero-allocation basic cell if needed
        if len(basic_cells) < required:
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in basic_cells:
                        trial = basic_cells + [(i, j)]
                        if find_closed_loop((i, j), basic_cells) is None:
                            basic_cells.append((i, j))
                            allocation[i][j] = 0  # epsilon ~ 0
                            break
                if len(basic_cells) == required:
                    break

        # Compute u_i, v_j potentials (u_1 = 0)
        u = [None] * rows
        v = [None] * cols
        u[0] = 0
        changed = True
        while changed:
            changed = False
            for (i, j) in basic_cells:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i][j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i][j] - v[j]
                    changed = True

        print(f"\n--- MODI Iteration {iteration} ---")
        print("u =", [f"{source_names[i]}:{u[i]}" for i in range(rows)])
        print("v =", [f"{dest_names[j]}:{v[j]}" for j in range(cols)])

        # Opportunity cost for non-basic cells
        opp_cost = {}
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in basic_cells:
                    opp_cost[(i, j)] = cost[i][j] - (u[i] + v[j])

        print("Opportunity costs (non-basic cells):")
        for (i, j), val in opp_cost.items():
            print(f"  {source_names[i]}-{dest_names[j]}: {val}")

        if not opp_cost or min(opp_cost.values()) >= 0:
            print("\nAll opportunity costs >= 0  ->  Optimal solution reached.")
            break

        # Entering cell = most negative opportunity cost
        enter = min(opp_cost, key=opp_cost.get)
        print(f"Entering cell: {source_names[enter[0]]}-{dest_names[enter[1]]} "
              f"(opportunity cost = {opp_cost[enter]})")

        loop = find_closed_loop(enter, basic_cells)
        print("Closed loop:", [f"{source_names[i]}-{dest_names[j]}" for (i, j) in loop])

        minus_cells = loop[1::2][:-1] if loop[-1] == loop[0] else loop[1::2]
        # loop returned as [start, ..., start]; drop duplicate end for processing
        loop_cells = loop[:-1]
        plus_cells = loop_cells[0::2]
        minus_cells = loop_cells[1::2]

        theta = min(allocation[i][j] for (i, j) in minus_cells)
        print(f"Theta (max units to reallocate) = {theta}")

        for (i, j) in plus_cells:
            allocation[i][j] += theta
        for (i, j) in minus_cells:
            allocation[i][j] -= theta

        iteration += 1

    return allocation


# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("TRANSPORTATION PROBLEM - INPUT DATA")
    print("=" * 70)
    print("Supply:", dict(zip(source_names, supply_orig)))
    print("Demand:", dict(zip(dest_names, demand_orig)))
    print("Cost matrix:")
    for i, row in enumerate(cost):
        print(f"  {source_names[i]}: {row}")

    print("\n" + "=" * 70)
    print("STEP 1: VOGEL'S APPROXIMATION METHOD (VAM) - Initial BFS")
    print("=" * 70)
    ibfs = vogel_approximation_method(supply_orig, demand_orig, cost)
    print_table(ibfs, cost, "Initial Basic Feasible Solution (VAM):")
    print(f"\nInitial (VAM) Total Transportation Cost = {total_cost(ibfs, cost)}")

    print("\n" + "=" * 70)
    print("STEP 2: MODI METHOD - Optimality Test & Improvement")
    print("=" * 70)
    optimal = modi_method(supply_orig, demand_orig, cost, ibfs)
    print_table(optimal, cost, "\nOptimal Allocation (MODI):")
    print(f"\nMinimum (Optimal) Total Transportation Cost = {total_cost(optimal, cost)}")