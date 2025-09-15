import numpy as np
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import random


class Worker:
    """Represents a worker with their attributes."""

    def __init__(self, id, true_cost, bid, reputation, group_id):
        self.id = id
        self.true_cost = true_cost
        self.bid = bid
        self.reputation = reputation
        self.group_id = group_id

    def __repr__(self):
        return f"Worker(id={self.id}, bid={self.bid:.2f}, rep={self.reputation:.2f}, group={self.group_id})"


class Requester:
    """Represents a requester with their budget and compatibility levels."""

    def __init__(self, id, budget, compatibility_levels):
        self.id = id
        self.budget = budget
        self.compatibility_levels = compatibility_levels  # dict: {group_id: tau}

    def __repr__(self):
        return f"Requester(id={self.id}, budget={self.budget:.2f})"


class SimulationEnvironment:
    """Manages the simulation setup and generation of problem instances."""

    def __init__(self, num_workers, num_requesters, num_groups, budget_range, cost_ranges, rep_range):
        self.num_workers = num_workers
        self.num_requesters = num_requesters
        self.num_groups = num_groups
        self.budget_range = budget_range
        self.cost_ranges = cost_ranges
        self.rep_range = rep_range
        self.workers = []
        self.requesters = []

    def generate_instance(self):
        """Generates a random instance of workers and requesters."""
        self.workers = []
        self.requesters = []

        # Generate workers
        for i in range(self.num_workers):
            group_id = np.random.randint(0, self.num_groups)
            # Assign cost based on data accuracy ranges from paper's Table I
            rand_choice = np.random.rand()
            if rand_choice < 1 / 3:  # Corresponds to (0.4, 0.6) accuracy
                cost_range = self.cost_ranges[0]
            elif rand_choice < 2 / 3:  # Corresponds to [0.6, 0.8) accuracy
                cost_range = self.cost_ranges[1]
            else:  # Corresponds to [0.8, 1.0] accuracy
                cost_range = self.cost_ranges[2]

            cost = np.random.uniform(cost_range[0], cost_range[1])
            reputation = np.random.uniform(self.rep_range[0], self.rep_range[1])

            # For simplicity, we assume workers bid truthfully in this simulation
            # as we are evaluating the mechanism's performance, not worker strategy.
            self.workers.append(Worker(id=i, true_cost=cost, bid=cost, reputation=reputation, group_id=group_id))

        # Generate requesters
        for j in range(self.num_requesters):
            budget = np.random.uniform(self.budget_range[0], self.budget_range[1])
            compatibility_levels = {}
            for l in range(self.num_groups):
                # Randomly assign tau_lj as per paper's description
                workers_in_group = sum(1 for w in self.workers if w.group_id == l)
                if workers_in_group > 0:
                    compatibility_levels[l] = np.random.randint(1, max(2, workers_in_group + 1))
                else:
                    compatibility_levels[l] = 1
            self.requesters.append(Requester(id=j, budget=budget, compatibility_levels=compatibility_levels))

        return self.workers, self.requesters


def solve_orp_with_max_flow(workers, requesters, groups):
    """
    Solves the Optimal Reputation Problem (ORP) using a max-flow formulation.
    Returns the maximum total reputation and the allocation.
    """
    if not workers:
        return 0, {}

    G = nx.DiGraph()
    source = 's'
    sink = 't'
    G.add_node(source)
    G.add_node(sink)

    worker_nodes = {w.id: f'w_{w.id}' for w in workers}
    requester_nodes = {r.id: f'r_{r.id}' for r in requesters}

    # Add worker nodes
    for w in workers:
        G.add_node(worker_nodes[w.id])

    # Add requester nodes
    for r in requesters:
        G.add_node(requester_nodes[r.id])

    # Create group-requester constraint nodes for proper flow control
    group_requester_nodes = {}
    for r in requesters:
        for group_id in groups:
            node_name = f'gr_{r.id}_{group_id}'
            group_requester_nodes[(r.id, group_id)] = node_name
            G.add_node(node_name)

    # Add edges from source to workers with capacity = reputation
    for w in workers:
        G.add_edge(source, worker_nodes[w.id], capacity=w.reputation)

    # Add edges from workers to group-requester nodes
    for w in workers:
        for r in requesters:
            gr_node = group_requester_nodes[(r.id, w.group_id)]
            # Edge capacity is infinite to allow flow
            G.add_edge(worker_nodes[w.id], gr_node, capacity=float('inf'))

    # Add edges from group-requester nodes to requesters with compatibility constraints
    for r in requesters:
        for group_id in groups:
            gr_node = group_requester_nodes[(r.id, group_id)]
            tau = r.compatibility_levels.get(group_id, 0)
            if tau > 0:
                # This edge enforces the compatibility constraint
                G.add_edge(gr_node, requester_nodes[r.id], capacity=tau)

    # Add edges from requesters to sink
    for r in requesters:
        G.add_edge(requester_nodes[r.id], sink, capacity=float('inf'))

    # Calculate max flow
    try:
        flow_value, flow_dict = nx.maximum_flow(G, source, sink)
    except nx.NetworkXError:
        return 0, {}

    # Determine allocation from flow
    allocation = {}
    for w in workers:
        allocation[w.id] = {}
        for r in requesters:
            gr_node = group_requester_nodes[(r.id, w.group_id)]
            # Check if there's flow from worker to group-requester node
            if worker_nodes[w.id] in flow_dict and gr_node in flow_dict[worker_nodes[w.id]]:
                flow_amount = flow_dict[worker_nodes[w.id]][gr_node]
                if flow_amount > 0.01:  # Threshold for floating point comparison
                    # Worker w is allocated to requester r
                    allocation[w.id][r.id] = 1

    return flow_value, allocation


def care_co(workers, requesters, total_budget):
    """Implements the CARE-CO mechanism (Algorithm 1)."""
    if not workers or not requesters:
        return {}, 0, 0

    # Sort workers by bid/reputation ratio
    sorted_workers = sorted(workers, key=lambda w: w.bid / w.reputation if w.reputation > 0 else float('inf'))

    groups = set(w.group_id for w in workers)

    k = -1
    M_k = 0

    # Find the critical worker s_k
    for i in range(len(sorted_workers)):
        current_workers_set = sorted_workers[:i + 1]
        M_i, _ = solve_orp_with_max_flow(current_workers_set, requesters, groups)

        if M_i > 0 and (sorted_workers[i].bid / sorted_workers[i].reputation) * M_i <= total_budget:
            k = i
            M_k = M_i
        else:
            break

    if k == -1:
        return {}, 0, 0

    # Winner set is S_k
    winner_pool = sorted_workers[:k + 1]

    # Final allocation based on S_k
    final_reputation, final_allocation_matrix = solve_orp_with_max_flow(winner_pool, requesters, groups)

    # Determine payment
    price_k_plus_1 = float('inf')
    if k + 1 < len(sorted_workers):
        w_k_plus_1 = sorted_workers[k + 1]
        if w_k_plus_1.reputation > 0:
            price_k_plus_1 = w_k_plus_1.bid / w_k_plus_1.reputation

    price_budget = total_budget / M_k if M_k > 0 else float('inf')

    clearing_price = min(price_k_plus_1, price_budget)

    payments = {}
    total_payment = 0
    selected_workers_set = set()

    for w in winner_pool:
        is_winner = False
        for r_id, allocated in final_allocation_matrix.get(w.id, {}).items():
            if allocated == 1:
                payment = w.reputation * clearing_price
                payments[w.id] = payment
                total_payment += payment
                is_winner = True
                selected_workers_set.add(w.id)
                break  # Worker assigned to at most one requester

    return payments, total_payment, final_reputation


def pea(workers, requesters, budgets):
    """Implements the PEA sub-mechanism (Algorithm 2)."""
    if not workers or not requesters:
        return {}, {}, 0

    # Sort workers by bid
    sorted_workers = sorted(workers, key=lambda w: w.bid)
    n = len(sorted_workers)

    # Generate virtual price set R_b
    R_b = {0}
    for r_id, budget in budgets.items():
        for t in range(1, n + 1):
            R_b.add(budget / t)
    sorted_R_b = sorted(list(R_b))

    # Find critical price r*
    r_star = float('inf')
    M_f_r_star = 0
    best_solution = None

    for r in sorted_R_b:
        if r == 0:
            continue

        # Employability E(r)
        E_r = sum(np.floor(budgets[b_id] / r) for b_id in budgets)

        # Available worker set S(r)
        S_r = [w for w in sorted_workers if w.bid <= r]

        if not S_r:
            continue

        # OSP sub-problem (Eq 5-9)
        prob_osp = pulp.LpProblem("OSP", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", ((w.id, req.id) for w in S_r for req in requesters), 0, 1, pulp.LpBinary)

        prob_osp += pulp.lpSum(x[w.id, req.id] for w in S_r for req in requesters)

        groups = set(w.group_id for w in S_r)
        for req in requesters:
            # Compatibility constraint
            for g_id in groups:
                workers_in_group = [w for w in S_r if w.group_id == g_id]
                if workers_in_group:
                    prob_osp += pulp.lpSum(
                        x[w.id, req.id] for w in workers_in_group) <= req.compatibility_levels.get(g_id, 1)
            # Employability constraint
            prob_osp += pulp.lpSum(x[w.id, req.id] for w in S_r) <= np.floor(budgets[req.id] / r)

        for w in S_r:
            # Each worker at most one requester
            prob_osp += pulp.lpSum(x[w.id, req.id] for req in requesters) <= 1

        prob_osp.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob_osp.status == pulp.LpStatusOptimal:
            M_f_r = pulp.value(prob_osp.objective)

            if M_f_r is not None and M_f_r >= E_r:
                r_star = r
                M_f_r_star = M_f_r
                best_solution = S_r
                break

    if r_star == float('inf') or best_solution is None:
        return {}, {}, 0

    # Winner selection with weighted minimization (Eq 10-15)
    S_r_star = best_solution

    prob_win = pulp.LpProblem("WinnerSelection", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", ((w.id, req.id) for w in S_r_star for req in requesters), 0, 1, pulp.LpBinary)

    # Assign weights w_i = 2^i (based on position in sorted order)
    weights = {}
    for i, w in enumerate(sorted_workers):
        weights[w.id] = 2 ** (i + 1)

    prob_win += pulp.lpSum(weights[w.id] * y[w.id, req.id] for w in S_r_star for req in requesters)

    # Add constraints similar to OSP, but with equality for total winners
    prob_win += pulp.lpSum(y[w.id, req.id] for w in S_r_star for req in requesters) == M_f_r_star

    groups = set(w.group_id for w in S_r_star)
    for req in requesters:
        for g_id in groups:
            workers_in_group = [w for w in S_r_star if w.group_id == g_id]
            if workers_in_group:
                prob_win += pulp.lpSum(
                    y[w.id, req.id] for w in workers_in_group) <= req.compatibility_levels.get(g_id, 1)
        prob_win += pulp.lpSum(y[w.id, req.id] for w in S_r_star) <= np.floor(budgets[req.id] / r_star)

    for w in S_r_star:
        prob_win += pulp.lpSum(y[w.id, req.id] for req in requesters) <= 1

    prob_win.solve(pulp.PULP_CBC_CMD(msg=0))

    winners = set()
    winner_allocations = {}
    if prob_win.status == pulp.LpStatusOptimal:
        for w in S_r_star:
            for req in requesters:
                if y[w.id, req.id].varValue and y[w.id, req.id].varValue > 0.5:
                    winners.add(w.id)
                    winner_allocations[w.id] = req.id

    # Payment scheme (simplified for simulation)
    payments = {w_id: r_star for w_id in winners}

    total_reputation = sum(w.reputation for w in workers if w.id in winners)

    return winner_allocations, payments, total_reputation


def care_no(workers, requesters, epsilon=10):
    """Implements the CARE-NO mechanism."""
    if not workers or not requesters:
        return {}, 0, 0

    v_min = min(w.reputation for w in workers) if workers else 1
    v_max = max(w.reputation for w in workers) if workers else 1

    rho_max = v_max / v_min
    gamma = int(np.ceil(np.log(rho_max) / np.log(epsilon))) if rho_max > 1 else 1

    # Partition workers into gamma sets
    partitions = [[] for _ in range(gamma)]
    for w in workers:
        rho_i = w.reputation / v_min
        if rho_i == 1:
            h = 0
        else:
            h = int(np.floor(np.log(rho_i) / np.log(epsilon)))
        if h < gamma:
            partitions[h].append(w)

    results = []
    budgets = {req.id: req.budget for req in requesters}

    for partition in partitions:
        if not partition:
            continue
        _, payments, reputation = pea(partition, requesters, budgets)

        results.append({'payments': payments, 'reputation': reputation})

    if not results:
        return {}, 0, 0

    # Sample one result
    final_result = random.choice(results)

    final_payments = final_result['payments']
    total_payment = sum(final_payments.values())
    total_reputation = final_result['reputation']

    return final_payments, total_payment, total_reputation


def rrafl(workers, requesters, total_budget):
    """Extended RRAFL baseline."""
    # RRAFL sorts by reputation/bid
    sorted_workers = sorted(workers, key=lambda w: w.reputation / w.bid if w.bid > 0 else float('inf'), reverse=True)

    selected_workers = []
    current_cost = 0

    for w in sorted_workers:
        if current_cost + w.bid <= total_budget:
            selected_workers.append(w)
            current_cost += w.bid
        else:
            break

    # Randomly assign winners to requesters without compatibility violation (simplified)
    total_reputation = sum(w.reputation for w in selected_workers)
    return {}, current_cost, total_reputation


def ranpri(workers, requesters, budgets):
    """RanPri baseline."""
    if not workers:
        return {}, 0, 0

    all_costs = [w.bid for w in workers]
    min_cost, max_cost = min(all_costs), max(all_costs)

    selected_workers = []
    remaining_budgets = {req.id: req.budget for req in requesters}

    for w in workers:
        price = random.uniform(min_cost, max_cost)
        if price >= w.bid:
            # Assign to a random requester with enough budget
            for req in random.sample(requesters, len(requesters)):
                if remaining_budgets[req.id] >= price:
                    # Simplified compatibility check
                    workers_in_group_for_req = sum(
                        1 for sw in selected_workers
                        if sw.group_id == w.group_id and hasattr(sw, 'assigned_req') and sw.assigned_req == req.id)
                    if workers_in_group_for_req < req.compatibility_levels.get(w.group_id, 1):
                        w.assigned_req = req.id
                        selected_workers.append(w)
                        remaining_budgets[req.id] -= price
                        break

    total_reputation = sum(w.reputation for w in selected_workers)
    total_payment = sum(w.bid for w in selected_workers)  # Payment is at least bid
    return {}, total_payment, total_reputation


def run_experiments():
    """Main function to run simulations and plot results."""
    # Parameters from the paper
    NUM_WORKERS = 120
    BUDGET_RANGE = [40, 80]
    COST_RANGES = [[2, 4], [3, 5], [4, 6]]
    REP_RANGE = [0.1, 1.0]
    NUM_GROUPS_FIXED = 10
    NUM_REQUESTERS_FIXED = 5

    requester_counts = range(2, 13, 2)
    group_counts = range(4, 25, 4)

    results_vs_requesters = {
        'CARE-CO': [], 'CARE-NO': [], 'RRAFL': [], 'RanPri': []
    }
    results_vs_groups = {
        'CARE-CO': [], 'CARE-NO': [], 'RRAFL': [], 'RanPri': []
    }

    # Experiment 1: Varying number of requesters
    print("Running experiment: varying number of requesters...")
    for num_req in requester_counts:
        env = SimulationEnvironment(NUM_WORKERS, num_req, NUM_GROUPS_FIXED, BUDGET_RANGE, COST_RANGES, REP_RANGE)
        workers, requesters = env.generate_instance()

        total_budget = sum(r.budget for r in requesters)
        budgets = {r.id: r.budget for r in requesters}

        _, _, rep_co = care_co(workers, requesters, total_budget)
        _, _, rep_no = care_no(workers, requesters)
        _, _, rep_rrafl = rrafl(workers, requesters, total_budget)
        _, _, rep_ranpri = ranpri(workers, requesters, budgets)

        results_vs_requesters['CARE-CO'].append(rep_co)
        results_vs_requesters['CARE-NO'].append(rep_no)
        results_vs_requesters['RRAFL'].append(rep_rrafl)
        results_vs_requesters['RanPri'].append(rep_ranpri)

    # Experiment 2: Varying number of groups
    print("Running experiment: varying number of groups...")
    for num_groups in group_counts:
        env = SimulationEnvironment(NUM_WORKERS, NUM_REQUESTERS_FIXED, num_groups, BUDGET_RANGE, COST_RANGES, REP_RANGE)
        workers, requesters = env.generate_instance()

        total_budget = sum(r.budget for r in requesters)
        budgets = {r.id: r.budget for r in requesters}

        _, _, rep_co = care_co(workers, requesters, total_budget)
        _, _, rep_no = care_no(workers, requesters)
        _, _, rep_rrafl = rrafl(workers, requesters, total_budget)
        _, _, rep_ranpri = ranpri(workers, requesters, budgets)

        results_vs_groups['CARE-CO'].append(rep_co)
        results_vs_groups['CARE-NO'].append(rep_no)
        results_vs_groups['RRAFL'].append(rep_rrafl)
        results_vs_groups['RanPri'].append(rep_ranpri)

    # Plotting results
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot vs. requesters
    for name, data in results_vs_requesters.items():
        linestyle = '-' if name in ['CARE-CO', 'CARE-NO'] else '--'
        axs[0].plot(requester_counts, data, label=name, linestyle=linestyle, marker='o')
    axs[0].set_title('Overall Reputation vs. # Requesters')
    axs[0].set_xlabel('# Requesters')
    axs[0].set_ylabel('Overall Reputation')
    axs[0].legend()
    axs[0].grid(True)

    # Plot vs. groups
    for name, data in results_vs_groups.items():
        linestyle = '-' if name in ['CARE-CO', 'CARE-NO'] else '--'
        axs[1].plot(group_counts, data, label=name, linestyle=linestyle, marker='o')
    axs[1].set_title('Overall Reputation vs. # Groups')
    axs[1].set_xlabel('# Groups')
    axs[1].set_ylabel('Overall Reputation')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("care_simulation_results.png")
    plt.show()
    print("Experiments finished. Plot saved to care_simulation_results.png")


if __name__ == '__main__':
    run_experiments()