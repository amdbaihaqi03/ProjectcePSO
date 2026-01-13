# ============================================================
# Streamlit Dashboard for PSO-CVRP
# Synced with Google Colab Result
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="PSO-CVRP Dashboard", layout="wide")

st.title("🚚 Particle Swarm Optimization for CVRP")
st.markdown("This dashboard runs **exactly same logic as Google Colab** to ensure equal Best Distance results.")

# ============================================================
# GLOBAL FIXED SEED (MUST MATCH COLAB)
# ============================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

rng = np.random.RandomState(GLOBAL_SEED)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("vrp_raw_dataset.csv")

data = load_data()
customers = data[data["node_type"] == "customer"].reset_index(drop=True)
coords = data[["x", "y"]].values
CAPACITY = 30

# ============================================================
# DISTANCE MATRIX
# ============================================================
@st.cache_data
def compute_distance_matrix(coords):
    N = len(coords)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i][j] = math.dist(coords[i], coords[j])
    return dist

distance_matrix = compute_distance_matrix(coords)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def decode_particle(position):
    order = np.argsort(position)
    routes, route, load = [], [0], 0

    for idx in order:
        cust_id = int(customers.loc[idx, "node_id"])
        demand = customers.loc[idx, "demand"]

        if load + demand <= CAPACITY:
            route.append(cust_id)
            load += demand
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust_id]
            load = demand

    route.append(0)
    routes.append(route)
    return routes


def total_distance(routes):
    return sum(
        distance_matrix[route[i]][route[i + 1]]
        for route in routes
        for i in range(len(route) - 1)
    )


def fitness(position):
    return total_distance(decode_particle(position))

# ============================================================
# PSO ALGORITHM
# ============================================================
def run_pso(num_particles, iterations, w, c1, c2):
    DIM = len(customers)

    particles = rng.rand(num_particles, DIM)
    velocities = np.zeros((num_particles, DIM))

    pbest = particles.copy()
    pbest_fit = np.array([fitness(p) for p in particles])

    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    convergence = []

    for _ in range(iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )

            particles[i] += velocities[i]
            fit = fitness(particles[i])

            if fit < pbest_fit[i]:
                pbest[i] = particles[i].copy()
                pbest_fit[i] = fit
                if fit < gbest_fit:
                    gbest, gbest_fit = particles[i].copy(), fit

        convergence.append(gbest_fit)

    return gbest, gbest_fit, convergence

# ============================================================
# PARAMETER TUNING (SAME AS COLAB)
# ============================================================
def evaluate_pso(num_particles, iterations, w, c1, c2, runs=2):
    best_distance = float("inf")
    best_position = None
    best_convergence = None
    distances = []

    for _ in range(runs):
        pos, dist, convergence = run_pso(num_particles, iterations, w, c1, c2)
        distances.append(dist)

        if dist < best_distance:
            best_distance = dist
            best_position = pos
            best_convergence = convergence

    return np.mean(distances), best_distance, best_position, best_convergence


def parameter_search():

    particle_list = [10, 20, 50]
    iteration_list = [50, 100, 200]
    w_list = [0.4, 0.6, 0.8]
    c1_list = [1.5, 2.0, 3.0]
    c2_list = [1.5, 2.0, 3.0]

    best_overall = float("inf")
    best_setting = None
    best_position = None
    best_convergence = None

    for npart in particle_list:
        for it in iteration_list:
            for w in w_list:
                for c1 in c1_list:
                    for c2 in c2_list:

                        avg_d, best_d, pos, convergence = evaluate_pso(
                            npart, it, w, c1, c2
                        )

                        if best_d < best_overall:
                            best_overall = best_d
                            best_setting = (npart, it, w, c1, c2)
                            best_position = pos
                            best_convergence = convergence

    return best_setting, best_overall, best_position, best_convergence

# ============================================================
# STREAMLIT UI
# ============================================================
st.sidebar.header("🎯 PSO Controls")
run_search = st.sidebar.button("Run Parameter Tuning (Same as Colab)")

if run_search:

    start = time.time()
    best_setting, best_distance, best_position, best_convergence = parameter_search()
    end = time.time()

    num_particles, iterations, w, c1, c2 = best_setting
    best_routes = decode_particle(best_position)

    st.success("Parameter Search Completed (Same as Colab)")
    st.write(f"Best Distance: **{best_distance:.4f}**")
    st.write(f"Runtime: **{end-start:.2f} seconds**")

    st.write("Best Parameters:")
    st.write(best_setting)

    fig_conv, ax = plt.subplots()
    ax.plot(best_convergence)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Distance")
    st.pyplot(fig_conv)

    st.subheader("Vehicle Routes")
    for i, r in enumerate(best_routes):
        st.write(f"Route {i+1}: {r}")
