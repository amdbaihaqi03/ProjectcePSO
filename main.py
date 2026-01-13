# ============================================================
# Streamlit Dashboard for PSO - CVRP
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import time

# ============================================================
# Random Seed
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# Dataset
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("vrp_raw_dataset.csv")

data = load_data()
customers = data[data["node_type"] == "customer"].reset_index(drop=True)
coords = data[["x", "y"]].values
CAPACITY = 30

# ============================================================
# Distance Matrix
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
# PSO
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
                    gbest = particles[i].copy()
                    gbest_fit = fit

        convergence.append(gbest_fit)

    return gbest, gbest_fit, convergence

# ============================================================
# Evaluation (Same As Colab)
# ============================================================
def evaluate_pso(num_particles, iterations, w, c1, c2, runs=2):
    best_distance = float("inf")
    best_position = None
    best_convergence = None

    for _ in range(runs):
        pos, dist, convergence = run_pso(num_particles, iterations, w, c1, c2)
        if dist < best_distance:
            best_distance = dist
            best_position = pos
            best_convergence = convergence

    return best_distance, best_position, best_convergence

# ============================================================
# Streamlit UI
# ============================================================
st.title("🚚 Particle Swarm Optimization for CVRP")
st.write("This dashboard reproduces **exact same behaviour as Google Colab**.")

# Sidebar Fixed Parameters (Same As Report)
st.sidebar.header("PSO Parameters (Fixed to Match Report)")
num_particles = st.sidebar.selectbox("Particles", [10,20,50], index=2)
iterations = st.sidebar.selectbox("Iterations", [50,100,200], index=0)
w = st.sidebar.selectbox("Inertia Weight", [0.4,0.6,0.8], index=0)
c1 = st.sidebar.selectbox("C1", [1.5,2.0,3.0], index=0)
c2 = st.sidebar.selectbox("C2", [1.5,2.0,3.0], index=0)

runs = st.sidebar.slider("Repetition (Like Colab evaluate)", 1,5,2)

if st.sidebar.button("Run PSO"):
    start = time.time()
    best_distance, position, convergence = evaluate_pso(num_particles, iterations, w, c1, c2, runs)
    runtime = time.time() - start
    routes = decode_particle(position)

    st.subheader("📊 Performance")
    col1,col2,col3 = st.columns(3)
    col1.metric("Best Distance", f"{best_distance:.4f}")
    col2.metric("Routes", len(routes))
    col3.metric("Runtime (s)", f"{runtime:.3f}")

    st.subheader("🚚 Routes")
    for i,r in enumerate(routes):
        st.write(f"Route {i+1}: {r}")

    st.subheader("📉 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(convergence)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Distance")
    st.pyplot(fig)

