import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Case 10 – Constraint-Aware Planner (Mini MPC Studio)",
    page_icon="🧩",
)

st.title("🧩 Case 10: The Constrained Corridor – Mini MPC Studio")
st.write(
    """
In this lab you build a **mini MPC planner** for a heated room.

- At each step the planner looks **N steps ahead**.  
- It tests several **candidate heater levels**.  
- It rejects candidates that **violate constraints**:  
  - temperature corridor \\([T_{min}, T_{max}]\\)  
  - maximum change in heater power per step.  
- It then chooses the candidate with the **lowest predicted cost**.

You can compare this **constraint-aware planner** with an
**unconstrained planner** that ignores the safety corridor.
"""
)

st.markdown("---")


# -----------------------------
# 1) System parameters
# -----------------------------
st.subheader("1️⃣ Room Heating System")

col_sys1, col_sys2, col_sys3 = st.columns(3)

with col_sys1:
    T_ambient = st.slider(
        "Ambient temperature T_amb (°C)",
        min_value=0.0,
        max_value=30.0,
        value=20.0,
        step=1.0,
    )
with col_sys2:
    T_set = st.slider(
        "Setpoint T_set (°C)",
        min_value=15.0,
        max_value=30.0,
        value=24.0,
        step=0.5,
    )
with col_sys3:
    tau = st.slider(
        "Time constant τ (s)",
        min_value=10.0,
        max_value=200.0,
        value=60.0,
        step=10.0,
        help="Larger τ → room responds more slowly.",
    )

k_heat = st.slider(
    "Heater gain k_heat",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Larger k_heat → same heater power heats the room faster.",
)

st.write(
    f"System: **T_amb = {T_ambient:.1f}°C**, **T_set = {T_set:.1f}°C**, "
    f"τ = **{tau:.0f} s**, k_heat = **{k_heat:.2f}**"
)


# -----------------------------
# 2) Simulation settings
# -----------------------------
st.subheader("2️⃣ Simulation Settings")

col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    t_max = st.slider(
        "Total time (s)",
        min_value=60.0,
        max_value=600.0,
        value=240.0,
        step=30.0,
    )
with col_sim2:
    dt = st.slider(
        "Time step Δt (s)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
    )

n_steps = int(t_max / dt) + 1
st.write(
    f"Simulation: **{t_max:.0f} s**, Δt = **{dt:.1f} s**, "
    f"steps ≈ **{n_steps}**"
)

col_init1, col_init2 = st.columns(2)
with col_init1:
    T_initial = st.slider(
        "Initial temperature T₀ (°C)",
        min_value=0.0,
        max_value=30.0,
        value=18.0,
        step=0.5,
    )
with col_init2:
    u_initial = st.slider(
        "Initial heater power u₀ (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
    )


# -----------------------------
# 3) Mini MPC planner parameters
# -----------------------------
st.subheader("3️⃣ Planner Parameters & Constraints")

col_mpc1, col_mpc2, col_mpc3 = st.columns(3)
with col_mpc1:
    N_pred = st.slider(
        "Prediction horizon N (steps)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many steps into the future the planner looks.",
    )
with col_mpc2:
    lambda_u = st.slider(
        "Control effort weight λ",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Larger λ punishes large heater power (energy saving).",
    )
with col_mpc3:
    u_step = st.slider(
        "Candidate heater step Δu (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Candidates: 0, Δu, 2Δu, ..., 100.",
    )

col_cstr1, col_cstr2, col_cstr3 = st.columns(3)
with col_cstr1:
    T_min = st.slider(
        "Temperature lower bound T_min (°C)",
        min_value=0.0,
        max_value=30.0,
        value=20.0,
        step=0.5,
    )
with col_cstr2:
    T_max = st.slider(
        "Temperature upper bound T_max (°C)",
        min_value=0.0,
        max_value=40.0,
        value=26.0,
        step=0.5,
    )
with col_cstr3:
    du_max = st.slider(
        "Max power change per step Δu_max (%)",
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=5.0,
        help="Safety limit: |u_k - u_{k-1}| ≤ Δu_max.",
    )

st.caption(
    "The **unconstrained planner** ignores T_min, T_max and Δu_max. "
    "The **constraint-aware planner** rejects candidate heater values that "
    "would break these limits."
)


# -----------------------------
# Helper: room model
# -----------------------------
def room_next_temperature(T_curr, T_amb, tau, k_heat, u, dt):
    """
    Simple room heating model:
        dT/dt = -(T - T_amb)/tau + k_heat * (u/100)
    """
    dTdt = -(T_curr - T_amb) / tau + k_heat * (u / 100.0)
    return T_curr + dTdt * dt


# -----------------------------
# Planner helpers
# -----------------------------
def simulate_future_constant_u(
    T_start, u, N_pred, dt, T_amb, tau, k_heat
):
    """Simulate N_pred steps ahead with constant heater power u."""
    T_tmp = T_start
    temps = []
    for _ in range(N_pred):
        T_tmp = room_next_temperature(T_tmp, T_amb, tau, k_heat, u, dt)
        temps.append(T_tmp)
    return np.array(temps)


def choose_u_unconstrained(
    T_curr, candidate_us, N_pred, dt, T_amb, tau, k_heat, T_set, lambda_u
):
    """Unconstrained look-ahead planner (no corridor, no rate limits)."""
    best_u = candidate_us[0]
    best_cost = float("inf")

    for u_cand in candidate_us:
        temps = simulate_future_constant_u(
            T_curr, u_cand, N_pred, dt, T_amb, tau, k_heat
        )
        tracking_cost = np.sum((temps - T_set) ** 2)
        energy_cost = lambda_u * (u_cand / 100.0) ** 2 * N_pred
        J = tracking_cost + energy_cost

        if J < best_cost:
            best_cost = J
            best_u = u_cand

    return best_u


def choose_u_constrained(
    T_curr,
    u_prev,
    candidate_us,
    N_pred,
    dt,
    T_amb,
    tau,
    k_heat,
    T_set,
    lambda_u,
    T_min,
    T_max,
    du_max,
):
    """
    Constraint-aware planner:
    - Rejects candidate u if |u - u_prev| > du_max
    - Rejects if predicted temperatures leave [T_min, T_max]
    - Among valid candidates, chooses lowest cost.
    - If all candidates invalid, falls back to 'closest' to u_prev.
    """
    best_u = None
    best_cost = float("inf")

    for u_cand in candidate_us:
        # Rate-of-change constraint
        if abs(u_cand - u_prev) > du_max:
            continue

        temps = simulate_future_constant_u(
            T_curr, u_cand, N_pred, dt, T_amb, tau, k_heat
        )

        # Temperature corridor constraint
        if np.any(temps < T_min) or np.any(temps > T_max):
            continue

        tracking_cost = np.sum((temps - T_set) ** 2)
        energy_cost = lambda_u * (u_cand / 100.0) ** 2 * N_pred
        J = tracking_cost + energy_cost

        if J < best_cost:
            best_cost = J
            best_u = u_cand

    if best_u is None:
        # All candidates violated constraints: fall back to closest to u_prev
        best_u = min(candidate_us, key=lambda u: abs(u - u_prev))

    return best_u


def simulate_planner_pair(
    T_amb,
    T_set,
    T0,
    u0,
    tau,
    k_heat,
    dt,
    n_steps,
    N_pred,
    lambda_u,
    T_min,
    T_max,
    du_max,
    u_step,
):
    """
    Simulate two planners:
      - unconstrained
      - constraint-aware
    """
    t = np.zeros(n_steps)
    T_uncon = np.zeros(n_steps)
    T_constr = np.zeros(n_steps)
    u_uncon = np.zeros(n_steps)
    u_constr = np.zeros(n_steps)

    T_uncon[0] = T0
    T_constr[0] = T0
    u_uncon[0] = u0
    u_constr[0] = u0

    candidate_us = np.arange(0.0, 100.0 + u_step, u_step)

    for k in range(n_steps - 1):
        # --- Unconstrained planner ---
        u_un = choose_u_unconstrained(
            T_uncon[k],
            candidate_us,
            N_pred,
            dt,
            T_amb,
            tau,
            k_heat,
            T_set,
            lambda_u,
        )

        # --- Constraint-aware planner ---
        u_co = choose_u_constrained(
            T_constr[k],
            u_constr[k],
            candidate_us,
            N_pred,
            dt,
            T_amb,
            tau,
            k_heat,
            T_set,
            lambda_u,
            T_min,
            T_max,
            du_max,
        )

        u_uncon[k] = u_un
        u_constr[k] = u_co

        # State update
        T_uncon[k + 1] = room_next_temperature(
            T_uncon[k], T_amb, tau, k_heat, u_un, dt
        )
        T_constr[k + 1] = room_next_temperature(
            T_constr[k], T_amb, tau, k_heat, u_co, dt
        )

        t[k + 1] = t[k] + dt

    # Copy last control for plotting
    u_uncon[-1] = u_uncon[-2]
    u_constr[-1] = u_constr[-2]

    return t, T_uncon, T_constr, u_uncon, u_constr


# -----------------------------
# 4) Run simulation
# -----------------------------
t, T_uncon, T_constr, u_uncon, u_constr = simulate_planner_pair(
    T_ambient,
    T_set,
    T_initial,
    u_initial,
    tau,
    k_heat,
    dt,
    n_steps,
    N_pred,
    lambda_u,
    T_min,
    T_max,
    du_max,
    u_step,
)

# -----------------------------
# 5) Plots
# -----------------------------
st.markdown("---")
st.subheader("4️⃣ Temperature vs Time – Unconstrained vs Constraint-Aware")

fig1, ax1 = plt.subplots(figsize=(7, 4))

# Corridor shading
ax1.fill_between(
    t,
    T_min,
    T_max,
    color="lightgray",
    alpha=0.3,
    label="Safe corridor [T_min, T_max]",
)

ax1.plot(t, T_uncon, label="Unconstrained planner T(t)")
ax1.plot(t, T_constr, label="Constraint-aware planner T(t)")
ax1.axhline(T_set, linestyle="--", color="black", label="Setpoint")

ax1.set_xlabel("t (s)")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Room Temperature Trajectories")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

st.pyplot(fig1)

st.subheader("Heater Power – Unconstrained vs Constraint-Aware")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, u_uncon, label="Unconstrained u(t)")
ax2.plot(t, u_constr, label="Constraint-aware u(t)")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("Heater power u (%)")
ax2.set_ylim(-5, 105)
ax2.set_title("Control Signals")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

st.pyplot(fig2)


# -----------------------------
# 6) Data table (first steps)
# -----------------------------
st.subheader("5️⃣ Sample Data Table (First 20 Steps)")

max_rows = min(20, n_steps)
df = pd.DataFrame(
    {
        "t (s)": t[:max_rows],
        "T_uncon": T_uncon[:max_rows],
        "u_uncon": u_uncon[:max_rows],
        "T_constr": T_constr[:max_rows],
        "u_constr": u_constr[:max_rows],
    }
)

st.dataframe(
    df.style.format(
        {
            "t (s)": "{:.1f}",
            "T_uncon": "{:.2f}",
            "u_uncon": "{:.2f}",
            "T_constr": "{:.2f}",
            "u_constr": "{:.2f}",
        }
    )
)


# -----------------------------
# 7) Teacher / discussion box
# -----------------------------
st.markdown("---")
with st.expander("👩‍🏫 Teacher Box – Constraints & Planning Intuition"):
    st.write(
        r"""
**Key ideas:**

- The planner evaluates future trajectories instead of reacting only to current error.
- Safety and comfort are encoded as **constraints**:
  - the room must stay inside a temperature corridor,
  - the heater cannot change too fast.
- The cost function balances **tracking the setpoint** and **saving energy**.

**Suggested questions:**

1. Compare the temperature plots:
   - When does the unconstrained planner violate the corridor?
   - How does the constraint-aware planner behave differently for the same N and λ?

2. Vary Δu_max:
   - What happens if Δu_max is very small?
   - What happens if there is effectively no rate limit (Δu_max ≈ 100%)?

3. Discuss real systems:
   - Give examples where ignoring constraints could be dangerous
     (chemical reactors, batteries, engines, medical devices…).
   - How could an MPC-like planner help in those systems?
"""
    )

st.caption(
    "Case 10 – Constraint-Aware Planner: a mini MPC studio that links "
    "future prediction, cost functions and safety constraints."
)
