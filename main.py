# ============================================================
# Streamlit Dashboard for PSO-CVRP
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
st.set_page_config(
    page_title="PSO-CVRP Dashboard",
    layout="wide"
)

st.title("🚚 Particle Swarm Optimization for CVRP")
st.markdown("Interactive dashboard to explore PSO performance and vehicle routing solutions.")

# ============================================================
# Set Random Seed
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# Load Dataset
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("vrp_raw_dataset.csv")

data = load_data()
customers = data[data['node_type'] == 'customer'].reset_index(drop=True)
coords = data[['x', 'y']].values
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
# Route Plot Function
# ============================================================
def plot_routes(routes, data):
    fig, ax = plt.subplots(figsize=(6, 4))  # slightly smaller

    depot = data[data['node_type'] == 'depot'].iloc[0]
    customers_plot = data[data['node_type'] == 'customer']

    # Customers (BLACK)
    ax.scatter(customers_plot['x'], customers_plot['y'],
               color='black', label="Customers")

    # Depot (YELLOW)
    ax.scatter(depot['x'], depot['y'],
               marker='s', s=100, color='yellow', label="Depot")

    # Routes
    for idx, route in enumerate(routes):
        xs, ys = [], []
        for node in route:
            row = data[data['node_id'] == node].iloc[0]
            xs.append(row['x'])
            ys.append(row['y'])

        ax.plot(
            xs, ys,
            marker='o',
            markerfacecolor='black',
            markeredgecolor='black',
            label=f"Route {idx+1}"
        )

    ax.set_title("Vehicle Routing Solution")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)

    return fig

# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 5, 50, 15)
iterations = st.sidebar.slider("Iterations", 20, 200, 80)
w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.4)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.5, 3.0, 2.0)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.5, 3.0, 2.0)

run_button = st.sidebar.button("🚀 Run PSO")

# ============================================================
# Run PSO
# ============================================================
if run_button:
    start_time = time.time()

    best_position, best_distance, convergence = run_pso(
        num_particles, iterations, w, c1, c2
    )

    runtime = time.time() - start_time
    best_routes = decode_particle(best_position)

    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Distance", f"{best_distance:.4f}")
    col2.metric("Routes Used", len(best_routes))
    col3.metric("Runtime (s)", f"{runtime:.3f}")
    col4.metric("Vehicle Capacity", CAPACITY)

    st.subheader("🚚 Vehicle Routes")
    for i, route in enumerate(best_routes):
        st.write(f"**Route {i+1}:** {route}")

    st.subheader("📉 Convergence Curve")
    fig_conv, ax = plt.subplots(figsize=(6, 4))
    ax.plot(convergence, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far Distance")
    ax.grid(True)
    st.pyplot(fig_conv)

    st.subheader("🗺️ Vehicle Route Visualization")
    fig_routes = plot_routes(best_routes, data)
    st.pyplot(fig_routes)

else:
    st.info("👈 Adjust PSO parameters and click **Run PSO** to start.")
