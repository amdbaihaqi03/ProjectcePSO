# ============================================================
# STREAMLIT Dashboard for PSO - VRP
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import time

# ============================================================
# Setup Page
# ============================================================
st.set_page_config(page_title="PSO - VRP Dashboard", layout="wide")
st.title("Particle Swarm Optimization for Vehicle Routing Problem (VRP)")
st.markdown("""
### Interactive Dashboard
Explore PSO performance, convergence, routing solution and parameter effect dynamically.
""")

# ============================================================
# Set Seed
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# Load Dataset
# ============================================================
@st.cache_data
def load_dataset():
    return pd.read_csv("vrp_raw_dataset.csv")

data = load_dataset()
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
# PSO Algorithm
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
# Plot Route
# ============================================================
def plot_routes(routes):
    fig, ax = plt.subplots(figsize=(6, 4))

    depot = data[data["node_type"] == "depot"].iloc[0]
    customers_plot = data[data["node_type"] == "customer"]

    ax.scatter(customers_plot["x"], customers_plot["y"], color="black", label="Customers")
    ax.scatter(depot["x"], depot["y"], color="red", marker="s", s=120, label="Depot")

    for idx, route in enumerate(routes):
        xs, ys = [], []
        for node in route:
            node_row = data[data["node_id"] == node].iloc[0]
            xs.append(node_row["x"])
            ys.append(node_row["y"])

        ax.plot(xs, ys, marker="o", label=f"Route {idx + 1}")

    ax.set_title("Vehicle Routing Solution")
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig


# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.header("PSO Parameters")

num_particles = st.sidebar.selectbox("Number of Particles", [10, 20, 50], index=2)
iterations = st.sidebar.selectbox("Iterations", [50, 100, 200], index=0)
w = st.sidebar.selectbox("Inertia Weight (w)", [0.4, 0.6, 0.8], index=0)
c1 = st.sidebar.selectbox("Cognitive Coefficient (c1)", [1.5, 2.0, 3.0], index=0)
c2 = st.sidebar.selectbox("Social Coefficient (c2)", [1.5, 2.0, 3.0], index=0)

run_button = st.sidebar.button("Run Optimization")

# ============================================================
# Run Optimization
# ============================================================
if run_button:

    start = time.time()
    best_position, best_distance, convergence = run_pso(num_particles, iterations, w, c1, c2)
    runtime = time.time() - start
    best_routes = decode_particle(best_position)

    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Distance", f"{best_distance:.4f}")
    col2.metric("Total Routes", len(best_routes))
    col3.metric("Runtime (seconds)", f"{runtime:.3f}")
    col4.metric("Vehicle Capacity", CAPACITY)

    st.subheader("Generated Routes")
    for i, route in enumerate(best_routes):
        st.write(f"**Route {i+1}:** {route}")

    st.subheader("Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(convergence)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far Distance")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Route Visualization")
    st.pyplot(plot_routes(best_routes))


else:
    st.info("Set parameter values and click **Run Optimization** to see results.")
