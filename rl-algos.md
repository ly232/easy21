# 🚀 Reinforcement Learning Algorithm Cheatsheet

A technical one-pager based on the **David Silver RL Course (UCL/DeepMind)**.

---

## 🟢 1. Model-Free Prediction (Value Estimation)
*Goal: Estimate the value function for a fixed policy.*

| Algorithm | Setting | Timing | Update Rule (Image) | Lecture |
| :--- | :--- | :--- | :--- | :--- |
| **MC Prediction** | Episodic | Offline | ![MC](https://latex.codecogs.com/svg.latex?V(S_t)%5Cleftarrow%20V(S_t)%20+%20%5Calpha%20%5BG_t%20-%20V(S_t)%5D) | L4 |
| **TD(0)** | Continuing | Online | ![TD](https://latex.codecogs.com/svg.latex?V(S_t)%5Cleftarrow%20V(S_t)%20+%20%5Calpha%20%5BR_%7Bt+1%7D%20+%20%5Cgamma%20V(S_%7Bt+1%7D)%20-%20V(S_t)%5D) | L4 |
| **Semi-Grad TD** | Linear Appx | Online | ![SG](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bw%7D%5Cleftarrow%5Cmathbf%7Bw%7D+%5Calpha%5BR+%5Cgamma%20%5Chat%7Bv%7D(S%27%2C%5Cmathbf%7Bw%7D)-%5Chat%7Bv%7D(S%2C%5Cmathbf%7Bw%7D)%5D%5Cnabla%5Chat%7Bv%7D(S%2C%5Cmathbf%7Bw%7D)) | L6 |

---

## 🔵 2. Model-Free Control (Policy Optimization)
*Goal: Find the optimal policy through interaction.*

| Algorithm | Policy | Target Logic (The "Lookahead") | Lecture |
| :--- | :--- | :--- | :--- |
| **Sarsa** | On-Policy | ![Sarsa](https://latex.codecogs.com/svg.latex?R%20+%20%5Cgamma%20Q(S%27%2C%20A%27)) | L5 |
| **Q-Learning** | Off-Policy | ![QL](https://latex.codecogs.com/svg.latex?R%20+%20%5Cgamma%20%5Cmax_a%20Q(S%27%2C%20a)) | L5 |
| **Expected Sarsa**| On-Policy | ![ES](https://latex.codecogs.com/svg.latex?R%20+%20%5Cgamma%20%5Csum_a%20%5Cpi(a%7CS%27)%20Q(S%27%2C%20a)) | L5 |

---

## 🟡 3. Model-Based & Planning
*Goal: Use environment dynamics $P$ and $R$ to compute values.*

| Algorithm | Usage | Equation | Lecture |
| :--- | :--- | :--- | :--- |
| **Value Iteration** | Control | ![VI](https://latex.codecogs.com/svg.latex?v_%7Bk+1%7D(s)%20%3D%20%5Cmax_a%20%5Csum_%7Bs%27%2Cr%7D%20p(s%27%2Cr%7Cs%2Ca)%5Br+%5Cgamma%20v_k(s%27)%5D) | L3 |
| **Policy Eval** | Prediction | ![PE](https://latex.codecogs.com/svg.latex?v_%7Bk+1%7D(s)%20%3D%20%5Csum_a%20%5Cpi(a%7Cs)%20%5Csum_%7Bs%27%2Cr%7D%20p(s%27%2Cr%7Cs%2Ca)%5Br+%5Cgamma%20v_k(s%27)%5D) | L3 |

---

## 🟠 4. The Bridge: Eligibility Traces
*Goal: Unify TD and MC via the weight $\lambda$.*

**Trace Update:** ![Trace](https://latex.codecogs.com/svg.latex?E_t(s)%20%3D%20%5Cgamma%5Clambda%20E_%7Bt-1%7D(s)%20+%20%5Cmathbb%7B1%7D(S_t%3Ds))  
**Mechanic:** Updates states based on frequency and recency. $\lambda=0$ is TD; $\lambda=1$ is MC.

---

## 🟣 5. Average Reward (Continuing Tasks)
*Goal: Maximize reward per time step in non-terminating environments.*

**TD Error:** ![AvgErr](https://latex.codecogs.com/svg.latex?%5Cdelta_t%20%3D%20R_%7Bt+1%7D%20-%20%5Cbar%7BR%7D_t%20+%20%5Chat%7Bv%7D(S_%7Bt+1%7D%2C%20%5Cmathbf%7Bw%7D)%20-%20%5Chat%7Bv%7D(S_t%2C%20%5Cmathbf%7Bw%7D))

| Algorithm | Policy Gradient Update | Lecture |
| :--- | :--- | :--- |
| **Softmax A-C** | ![Softmax](https://latex.codecogs.com/svg.latex?%5Cdelta_t%20%5Cnabla_%5Ctheta%20%5Cln%20%5Cpi(A_t%7CS_t%2C%20%5Ctheta)) | L7 |
| **Gaussian A-C** | ![Gaussian](https://latex.codecogs.com/svg.latex?%5Cdelta_t%20%5Cnabla_%5Ctheta%20%5Cln%20%5Cmathcal%7BN%7D(A_t%7C%5Cmu%2C%5Csigma)) | L7 |

---

## 📉 6. Convergence Properties

| Algorithm | Tabular | Linear Function Appx | David Silver |
| :--- | :--- | :--- | :--- |
| **DP** | Converges to $v_*$ | N/A | L3 |
| **MC** | Converges to $v_*$ | Converges to Min MSE | L4 |
| **TD / Sarsa** | Converges to $v_*$ | Converges to TD Fixed Point | L4/L6 |
| **Q-Learning** | Converges to $q_*$ | **May Diverge** (Deadly Triad) | L5/L6 |

---

## 🧠 Core Definitions
* **On-Policy:** Learn about the policy you are executing (Sarsa, AC).
* **Off-Policy:** Learn about optimal policy while exploring (Q-Learning).
* **Bootstrapping:** Updating an estimate using an estimate ($R + \gamma \hat{V}$).
* **The Deadly Triad:** Function Approximation + Bootstrapping + Off-Policy.
