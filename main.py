# ============================================================
# Streamlit Dashboard for PSO-CVRP (Consistent with Colab)
# Course: JIE42903 – Evolutionary Computing
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import time

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(page_title="PSO-CVRP Dashboard", layout="wide")
st.title("🚚 Particle Swarm Optimization for CVRP")
st.markdown("This Streamlit implementation is synchronized with the Colab version.")

# ============================================================
# Load Dataset (NO RANDOMNESS HERE)
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("vrp_raw_dataset.csv")

data = load_data()
customers = data[data["node_type"] == "customer"].reset_index(drop=True)
coords = data[["x", "y"]].values
CAPACITY = 30

# ============================================================
# Distance Matrix (DETERMINISTIC)
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
# Helper Functions (SAME AS COLAB)
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
# PSO Algorithm (IDENTICAL TO COLAB)
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
# Route Visualization
# ============================================================

def plot_routes(routes):
    fig, ax = plt.subplots(figsize=(6, 4))

    depot = data[data["node_type"] == "depot"].iloc[0]
    cust = data[data["node_type"] == "customer"]

    ax.scatter(cust["x"], cust["y"], color="black", label="Customers")
    ax.scatter(depot["x"], depot["y"], marker="s", s=100, color="red", label="Depot")

    for i, route in enumerate(routes):
        xs, ys = [], []
        for node in route:
            row = data[data["node_id"] == node].iloc[0]
            xs.append(row["x"])
            ys.append(row["y"])
        ax.plot(xs, ys, marker="o", label=f"Route {i+1}")

    ax.set_title("Best PSO Routing Solution")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(fontsize=8)
    ax.grid(True)

    return fig

# ============================================================
# Sidebar Controls (MATCH COLAB PARAMETERS)
# ============================================================

st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.selectbox("Number of Particles", [10, 20, 50], index=0)
iterations = st.sidebar.selectbox("Iterations", [50, 100, 200], index=0)
w = st.sidebar.selectbox("Inertia Weight (w)", [0.4, 0.6, 0.8], index=0)
c1 = st.sidebar.selectbox("Cognitive Coefficient (c1)", [1.5, 2.0, 3.0], index=0)
c2 = st.sidebar.selectbox("Social Coefficient (c2)", [1.5, 2.0, 3.0], index=0)

run_button = st.sidebar.button("🚀 Run PSO")

# ============================================================
# Run PSO (RESET SEED HERE ❗❗❗)
# ============================================================

if run_button:

    # 🔑 THIS IS THE KEY FIX
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    start_time = time.time()

    best_position, best_distance, convergence = run_pso(
        num_particles, iterations, w, c1, c2
    )

    runtime = time.time() - start_time
    best_routes = decode_particle(best_position)

    # ========================================================
    # Performance Metrics
    # ========================================================

    st.subheader("📊 Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Distance", f"{best_distance:.4f}")
    col2.metric("Routes Used", len(best_routes))
    col3.metric("Runtime (s)", f"{runtime:.3f}")

    # ========================================================
    # Routes Output
    # ========================================================

    st.subheader("🚚 Vehicle Routes")
    for i, route in enumerate(best_routes):
        st.write(f"**Vehicle {i+1}:** {route}")

    # ========================================================
    # Convergence Curve
    # ========================================================

    st.subheader("📉 Convergence Curve")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(convergence, linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best-so-far Distance")
    ax1.grid(True)
    st.pyplot(fig1)

    # ========================================================
    # Route Visualization
    # ========================================================

    st.subheader("🗺️ Route Visualization")
    fig2 = plot_routes(best_routes)
    st.pyplot(fig2)

else:
    st.info("👈 Set parameters and click **Run PSO**")
