# ============================================================
# Streamlit Dashboard for PSO-VRP (FIXED VERSION)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="PSO-VRP Dashboard", layout="wide")
st.title("Particle Swarm Optimization for VRP")

# ============================================================
# Load Dataset (OK to cache)
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("vrp_raw_dataset.csv")

data = load_data()
customers = data[data['node_type'] == 'customer'].reset_index(drop=True)
coords = data[['x', 'y']].values
CAPACITY = 30

# ============================================================
# Distance Matrix (OK to cache)
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
# Helper Functions
# ============================================================
def decode_particle(position):
    order = np.argsort(position)
    routes, route, load = [], [0], 0

    for idx in order:
        cust_id = int(customers.loc[idx, 'node_id'])
        demand = customers.loc[idx, 'demand']

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
# PSO Algorithm (UNCHANGED)
# ============================================================
def run_pso(num_particles, iterations, w, c1, c2):
    DIM = len(customers)
    particles = np.random.rand(num_particles, DIM)
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
# Sidebar Controls
# ============================================================
st.sidebar.header("PSO Parameters")

num_particles = st.sidebar.slider("Particles", 10, 100, 50)
iterations = st.sidebar.slider("Iterations", 20, 200, 100)
w = st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.4)
c1 = st.sidebar.slider("Cognitive (c1)", 0.5, 3.0, 2.0)
c2 = st.sidebar.slider("Social (c2)", 0.5, 3.0, 2.0)

run_button = st.sidebar.button("Run PSO")

# ============================================================
# Run PSO (SEED RESET HERE!)
# ============================================================
if run_button:

    # 🔥 IMPORTANT FIX 🔥
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    start_time = time.time()

    best_position, best_distance, convergence = run_pso(
        num_particles, iterations, w, c1, c2
    )

    runtime = time.time() - start_time
    best_routes = decode_particle(best_position)

    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Distance", f"{best_distance:.4f}")
    col2.metric("Routes Used", len(best_routes))
    col3.metric("Runtime (s)", f"{runtime:.3f}")

    st.subheader("Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(convergence)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far Distance")
    ax.grid(True)
    st.pyplot(fig)
