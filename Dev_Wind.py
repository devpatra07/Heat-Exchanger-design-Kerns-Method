
import streamlit as st
import math

st.set_page_config(page_title="Heat Exchanger Design - Kern’s Method", layout="wide")

st.title("Heat Exchanger Design using Kern’s Method")
st.markdown("**Created by Dev Patra** — Still in developing phase. Visit: [**https://devpatra07.github.io/**](https://devpatra07.github.io/)")

st.sidebar.header("Input Parameters")

# Fluid properties
m_dot_hot = st.sidebar.number_input("Hot fluid mass flow rate (kg/s)", value=1.5)
Cp_hot = st.sidebar.number_input("Hot fluid specific heat (kJ/kg·K)", value=4.18)
T_hot_in = st.sidebar.number_input("Hot fluid inlet temperature (°C)", value=150.0)
T_hot_out = st.sidebar.number_input("Hot fluid outlet temperature (°C)", value=100.0)

m_dot_cold = st.sidebar.number_input("Cold fluid mass flow rate (kg/s)", value=2.0)
Cp_cold = st.sidebar.number_input("Cold fluid specific heat (kJ/kg·K)", value=4.18)
T_cold_in = st.sidebar.number_input("Cold fluid inlet temperature (°C)", value=30.0)
T_cold_out = st.sidebar.number_input("Cold fluid outlet temperature (°C)", value=70.0)

# Geometry and coefficients
U = st.sidebar.number_input("Overall heat transfer coefficient U (W/m²·K)", value=600.0)
fouling_factor = st.sidebar.number_input("Fouling factor (m²·K/W)", value=0.0002)
tube_length = st.sidebar.number_input("Tube length (m)", value=3.0)
tube_outer_diameter = st.sidebar.number_input("Tube outer diameter (m)", value=0.025)
tube_inner_diameter = st.sidebar.number_input("Tube inner diameter (m)", value=0.02)

# Calculate heat duty
Q_hot = m_dot_hot * Cp_hot * (T_hot_in - T_hot_out) * 1000  # W
Q_cold = m_dot_cold * Cp_cold * (T_cold_out - T_cold_in) * 1000  # W
Q_avg = (Q_hot + Q_cold) / 2

# Log Mean Temperature Difference (LMTD)
delta_T1 = T_hot_in - T_cold_out
delta_T2 = T_hot_out - T_cold_in

if delta_T1 == delta_T2:
    LMTD = delta_T1
else:
    LMTD = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)

# Corrected U for fouling
U_cleaned = 1 / ((1 / U) + fouling_factor)

# Required surface area
A_required = Q_avg / (U_cleaned * LMTD)

# Number of tubes
tube_area = math.pi * tube_outer_diameter * tube_length
N_tubes = A_required / tube_area

# Output
st.subheader("📋 Design Results")
st.write(f"**Heat Duty (Average):** {Q_avg/1000:.2f} kW")
st.write(f"**Log Mean Temperature Difference (LMTD):** {LMTD:.2f} °C")
st.write(f"**Corrected Overall Heat Transfer Coefficient (U):** {U_cleaned:.2f} W/m²·K")
st.write(f"**Required Heat Transfer Area:** {A_required:.2f} m²")
st.write(f"**Estimated Number of Tubes:** {math.ceil(N_tubes)}")

with st.expander("🔍 Assumptions & Notes"):
    st.markdown("""
    - Counter-current flow assumed.
    - Heat balance checked using average Q from both streams.
    - LMTD is used for area calculation.
    - Tubes assumed to be uniformly sized with no extended surfaces.
    """)

st.markdown("---")
st.caption("Developed by Dev Patra | Kern’s Method Based Design | Streamlit")