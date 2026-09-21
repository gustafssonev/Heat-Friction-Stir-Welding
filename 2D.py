# ==================================================================
# Preparation
# ==================================================================

#imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches

# device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

# results saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results") / f"pinn_scaled_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)
print(f"Results saved to: {results_dir}")

# ==================================================================
# Physics
# ==================================================================

# physical values
alpha = 4.564e-5        # thermal diffusivity (m²/s) for aluminum
k = 112.0               # thermal conductivity (W/m·K) for aluminum
T_room = 20.0           # room temperature (°C)
T_melt = 660.0          # melting point of aluminum (°C)
T_steel = 80
theta_steel = (T_steel - T_room)/525
h_air = 25              # convection coefficient for air (W/m²·K) 
h_steel = 1000          # convection coefficient for steel (W/m²·K) 
''' check rho and cp values'''
rho = 2700.0            # density (kg/m³) for aluminum
cp = 900.0              # specific heat capacity (J/kg·K) for aluminum
eps = 1e-8

# domain
a1, b1 = 0.0, 1.04775e-1 # x-domain (m)
a2, b2 = 0.0, 6.35e-3    # y-domain (m)
t_end = 210.0            # end time (s)
Rsh_raw = 0.01905        # shoulder radius (m) 
Rp_raw = 0.00635              # pin radius (m)
Hp_raw = 0.0047752            # pin height (m)
x_center_raw = (a1 + b1) / 2  # center x coordinate
shoulder_left_x = [(x_center_raw - Rsh_raw), (x_center_raw - Rsh_raw)]
shoulder_right_x = [(x_center_raw + Rsh_raw), (x_center_raw + Rsh_raw)]
shoulder_y = [a2, b2]
pin_width = 2 * Rp_raw
pin_left = x_center_raw - Rp_raw
pin_right = x_center_raw + Rp_raw
pin_bottom = b2 - Hp_raw

# Dimensionless domain
L = b2 - a2                     # length (m)
t_c = L**2 / alpha              # time scale (s)
x_min, x_max = a1 / L, b1 / L   # scaled x domain
y_min, y_max = a2 / L, b2 / L   # scaled y domain
t_min, t_max = 0.0, t_end / t_c # scaled time domain
coef_air = h_air * L / k        # biot number for convection BC
# coef_steel = h_steel * L / k    # biot number for bottom 
coef_steel = 1
Rsh = Rsh_raw / L               # scaled shoulder radius
Rp = Rp_raw / L
Hp = Hp_raw / L
x_center = (x_min + x_max) / 2  # center x-coordinate

# T vs RPM data (from Dr.S)
exp_rpm = np.array([410, 360, 310, 410, 360, 310, 330, 280, 230], dtype=float)
exp_temp = np.array([543, 454, 347, 507, 429, 327, 447, 367, 273], dtype=float)

# For heat input scaling
deltaT_exp = exp_temp - T_room           # expected temperature rise above room temp
q_exp = k * deltaT_exp / L               # expected heat flux

# Input rpm we want to test, also used as reference
rpm_ref = 410.0                                       # rpm (highest experimental rpm reading)
exp_T_at_ref = np.mean(exp_temp[exp_rpm == rpm_ref])  # corresponding experimental temperature at that rpm
q_ref = k * (exp_T_at_ref - T_room) / L               # reference heat flux
DeltaT_ref = exp_T_at_ref - T_room                    # reference temp rise
DeltaT_r = q_ref * L / k                              # reference dimensionless temperature rise

# Scaling flux 
flux_exp_dimless = q_exp / q_ref                                          # dimensionless flux
a_flux_dimless, b_flux_dimless = np.polyfit(exp_rpm, flux_exp_dimless, 1) # linear fit
a_flux_dimless = float(a_flux_dimless)                                    # float for PyTorch compatibility
b_flux_dimless = float(b_flux_dimless)                                    # float for PyTorch compatibility

# get radius of rotation
def get_radius(x):
    r = x - x_center
    return r

# get velocity of rotation
def get_velocity(rpm_tensor, radius):
    pi = 3.1415926535
    v = radius * (rpm_tensor * 2 * pi) / 60.0
    return v

# get Q: internal heat from pin
def get_internal_source(x, y, rpm_tensor):
    r_local = torch.abs(get_radius(x))               # radial distance from center
    k_s = 200.0                                       # controls how sharp the edge of the pin is (for better gradients)
    mask_x = torch.sigmoid(k_s * (Rp - r_local))     # 1 inside pin radius, 0 outside, with smooth transition
    mask_y = torch.sigmoid(k_s * (y - (y_max - Hp))) # 1 above pin height, 0 below, with smooth transition
    combined_mask = mask_x * mask_y                  # 1 in pin region, 0 elsewhere, with smooth edges

    q_magnitude = (a_flux_dimless * rpm_tensor + b_flux_dimless) 
    # Q = q_magnitude * (r_local / Rp) * combined_mask
    Q = q_magnitude * combined_mask
    
    return Q

# Output for debugging purposes
test_rpms = np.array([230, 280, 310, 330, 360, 410])  # all rpms ordered
for r in test_rpms:                                   # output flux ratio at each rpm
    fr = a_flux_dimless * r + b_flux_dimless
    print(f"RPM {r}: flux_ratio = {fr:.4f}")
print(f"q_ref (W/m²): {q_ref:.2f}")                                                     # output reference heat flux
print(f"DeltaT_ref (°C): {DeltaT_ref:.2f}")                                             # output reference temperature rise
print(f"Flux ratio coefficients: a={a_flux_dimless:.6f}, b={b_flux_dimless:.6f}")       # output linear fit coefficients
print(f"flux_ratio(reference RPM) = {a_flux_dimless * rpm_ref + b_flux_dimless:.6f}")   # output flux ratio at reference RPM
print(f"Expected T_max at reference RPM: {T_room + DeltaT_ref:.2f}°C")                  # output expected max temperature at reference RPM 

# ==================================================================
# Collocation Points
# ==================================================================

Nf, Ni, Nb, Nb_ps = 2000, 500, 1200, 1200 # number of collocation points for PDE, initial condition, and boundary conditions

# pde
x_f = torch.rand(Nf,1,device=device)*(x_max-x_min)+x_min
y_f = torch.rand(Nf,1,device=device)*(y_max-y_min)+y_min
t_f = torch.rand(Nf,1,device=device)*(t_max-t_min)+t_min
x_f.requires_grad_(True)
y_f.requires_grad_(True)
t_f.requires_grad_(True)

# initial
x_i = torch.rand(Ni,1,device=device)*(x_max-x_min)+x_min
y_i = torch.rand(Ni,1,device=device)*(y_max-y_min)+y_min
t_i = torch.zeros(Ni,1,device=device, requires_grad=True)

# bottom
x_b = torch.rand(Nb,1,device=device)*(x_max-x_min)+x_min
y_b = torch.full((Nb,1), y_min, device=device)
t_b = torch.rand(Nb,1,device=device)*(t_max-t_min)+t_min
x_b.requires_grad_(True)
y_b.requires_grad_(True)
t_b.requires_grad_(True)

# top
x_t_rand = torch.rand(Nb//2, 1, device=device) * (x_max - x_min) + x_min
x_t_tool = torch.rand(Nb//2, 1, device=device) * (2 * Rsh) + (x_center - Rsh)
x_t = torch.cat([x_t_rand, x_t_tool], dim=0)
y_t = torch.full((Nb,1), y_max, device=device)
t_t = torch.rand(Nb,1,device=device)*(t_max-t_min)+t_min
x_t.requires_grad_(True)
y_t.requires_grad_(True)
t_t.requires_grad_(True)

x_ps_left = torch.full((Nb_ps//2, 1), pin_left, device=device)
x_ps_right = torch.full((Nb_ps//2, 1), pin_right, device=device)
x_ps = torch.cat([x_ps_left, x_ps_right], dim=0)
y_ps = torch.rand(Nb_ps, 1, device=device) * (b2 - pin_bottom) + pin_bottom
t_ps = torch.rand(Nb_ps, 1, device=device) * t_end
x_ps.requires_grad_(True)
y_ps.requires_grad_(True)
t_ps.requires_grad_(True)

# left
x_l = torch.full((Nb,1), x_min, device=device)
y_l = torch.rand(Nb,1,device=device)*(y_max-y_min)+y_min
t_l = torch.rand(Nb,1,device=device)*(t_max-t_min)+t_min
x_l.requires_grad_(True)
y_l.requires_grad_(True)
t_l.requires_grad_(True)

# right
x_r = torch.full((Nb,1), x_max, device=device)
y_r = torch.rand(Nb,1,device=device)*(y_max-y_min)+y_min
t_r = torch.rand(Nb,1,device=device)*(t_max-t_min)+t_min
x_r.requires_grad_(True)
y_r.requires_grad_(True)
t_r.requires_grad_(True)

# ==================================================================
# Model
# ==================================================================

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(4, 64), nn.SiLU()]
        for _ in range(8):
            layers += [nn.Linear(64, 64), nn.SiLU()]
        layers.append(nn.Linear(64, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t, rpm):
        x_half_width = (x_max - x_min) / 2
        x_norm = (x - x_center) / x_half_width
        y_norm = y / y_max
        t_norm = t / t_max
        rpm_norm = (rpm - 150.0) / (450.0 - 150.0)
        
        out = self.net(torch.cat([x_norm, y_norm, t_norm, rpm_norm], dim=1))
        base_theta = torch.nn.functional.softplus(out) # softplus for > 0 output
        flux_ratio = (a_flux_dimless * rpm + b_flux_dimless)
        theta = base_theta * flux_ratio
        return theta
    
# Instantiate the network
net = PINN().to(device)

# ==================================================================
# Loss Function
# ==================================================================

def pinn_loss(rpm_tensor, iteration, bc_top_multiplier=1.0):

    if rpm_tensor.dim() == 1:
        rpm_tensor = rpm_tensor.unsqueeze(1)

    rpm_f = rpm_tensor.repeat(Nf, 1)

    theta = net(x_f, y_f, t_f, rpm_f)

    # derivatives
    theta_t = torch.autograd.grad(
        theta, t_f, torch.ones_like(theta), create_graph=True
    )[0]

    theta_x = torch.autograd.grad(
        theta, x_f, torch.ones_like(theta), create_graph=True
    )[0]

    theta_y = torch.autograd.grad(
        theta, y_f, torch.ones_like(theta), create_graph=True
    )[0]

    theta_xx = torch.autograd.grad(
        theta_x, x_f, torch.ones_like(theta_x), create_graph=True
    )[0]

    theta_yy = torch.autograd.grad(
        theta_y, y_f, torch.ones_like(theta_y), create_graph=True
    )[0]

    # heat equation
    # mse_pde = torch.mean((theta_t - (theta_xx + theta_yy)) ** 2)
    Q_internal = get_internal_source(x_f, y_f, rpm_f)
    Q_internal_scaled = (Q_internal * L**2) / (k * DeltaT_ref)             # scale internal source by reference flux

    v_x = get_velocity(rpm_f, get_radius(x_f))
    v_x_scaled = v_x * L / alpha                                           # scale velocity
    
    # residual = (theta_t + v_x_scaled * theta_x) - (theta_xx + theta_yy) - Q_internal_scaled
    residual = (theta_t) - (theta_xx + theta_yy) - Q_internal_scaled
    res_squared = residual ** 2

    r_f = torch.abs(get_radius(x_f))
    is_pin = (r_f <= Rp) & (y_f >= (y_max - Hp)) # within radius and below the top surface by height Hp

    # Split loss
    if is_pin.any():
        loss_pde_pin = torch.mean(res_squared[is_pin])
    else:
        loss_pde_pin = torch.tensor(0.0, device=device)

    if (~is_pin).any():
        loss_pde_bulk = torch.mean(res_squared[~is_pin])
    else:
        loss_pde_bulk = torch.tensor(0.0, device=device)

    mse_pde = loss_pde_pin + loss_pde_bulk # just for print
 
    # Initial condition: θ(x,y,0) = 0
    theta_ic = net(x_i, y_i, t_i, rpm_tensor.repeat(Ni, 1))
    mse_ic = torch.mean(theta_ic ** 2)

    # Bottom BC: ∂θ/∂y = conv
    theta_bottom = net(x_b, y_b, t_b, rpm_tensor.repeat(Nb, 1))

    theta_yb = torch.autograd.grad(
        theta_bottom, y_b,
        torch.ones_like(theta_bottom),
        create_graph=True
    )[0]

    # mse_bottom = torch.mean(theta_yb ** 2)
    mse_bottom = torch.mean((theta_yb - coef_steel * (theta_bottom - theta_steel))**2)

    # Left BC: ∂θ/∂x = hLθ/k
    theta_left = net(x_l, y_l, t_l, rpm_tensor.repeat(Nb, 1))

    theta_xl = torch.autograd.grad(
        theta_left, x_l,
        torch.ones_like(theta_left),
        create_graph=True
    )[0]

    mse_left = torch.mean((theta_xl - coef_air * theta_left) ** 2)

    # Right BC: ∂θ/∂x = -hLθ/k
    theta_right = net(x_r, y_r, t_r,rpm_tensor.repeat(Nb, 1))

    theta_xr = torch.autograd.grad(
        theta_right, x_r,
        torch.ones_like(theta_right),
        create_graph=True
    )[0]

    mse_right = torch.mean((theta_xr + coef_air * theta_right) ** 2)

    # Top BC: RPM-scaled heat flux
    theta_top = net(x_t, y_t, t_t, rpm_tensor.repeat(Nb, 1))

    theta_yt = torch.autograd.grad(
        theta_top, y_t,
        torch.ones_like(theta_top),
        create_graph=True
    )[0]

    flux_ratio = (a_flux_dimless * rpm_tensor + b_flux_dimless) # dimensionless flux ratio at current RPM
    flux_target = flux_ratio.expand_as(theta_yt)                # target dimensionless flux at each collocation point on the top boundary

    r_shoulder_heat = torch.abs(x_t - x_center)                
    mask = ((r_shoulder_heat >= Rp) & (r_shoulder_heat <= Rsh)).squeeze() # shoulder diameter mask
    
    # Separate losses for heated shoulder region and convective outer region
    if mask.any():
        loss_heat = torch.mean((theta_yt[mask] - flux_target[mask])**2)
    else:
        loss_heat = torch.tensor(0.0, device=device, requires_grad=True)

    if (~mask).any():
        loss_conv = torch.mean((theta_yt[~mask] + coef_air * theta_top[~mask])**2)
    else:
        loss_conv = torch.tensor(0.0, device=device, requires_grad=True)

    theta_side = net(x_ps, y_ps, t_ps, rpm_tensor.repeat(Nb_ps, 1))

    theta_x_side = torch.autograd.grad(theta_side, x_ps, torch.ones_like(theta_side), create_graph=True)[0]
    mask_side = torch.ones_like(x_ps, dtype=torch.bool).squeeze()
    flux_target_side = torch.zeros_like(theta_x_side)
    loss_pin_side = torch.mean((theta_x_side[mask_side] - flux_target_side[mask_side])**2)

    # melt point
    theta_melt_limit = (T_melt - T_room) / DeltaT_r
    melt_penalty = torch.mean(torch.relu(theta - theta_melt_limit)**2)

    # for loss weighting
    transition = min(1.0, iteration / 800.0)
    wBC_top_base = 2000 if iteration < 500 else 1000

    # Loss weights
    # wPDE = 1.0 + (199.0 * transition)  # Start with 1.0, increase to 200.0
    # wIC = 2.0 + (10 * transition)    # Start with 2.0, then increase to 12.0
    wIC = 1000.0  # Keep initial condition weight high to ensure it is learned well
    wBC_bottom = 10
    wBC_left = 10
    wBC_right = 10
    # wBC_top = 1000  
    wBC_top_heat = wBC_top_base * bc_top_multiplier # 2000 if iteration < 500 else 1000
    wBC_top_cold = 500
    wMelt = 5000.0  
    wPDE_bulk = (1.0 + (199.0 * transition)) if iteration < 1000 else 1000 # Start with 1.0, increase to 200.0, 1000 for lbfgs
    wPDE_pin = (1.0 + (199.0 * transition)) if iteration < 1000 else 1000 # Start with 1.0, increase to 200.0, 1000 for lbfgs

    # Debugging prints
    if iteration % 100 == 0:
        print(
            f"loss_pde={mse_pde.item():.3e} "
            f"loss_bc_top={loss_heat.item():.3e} "
            f"loss_bc_bottom={mse_bottom.item():.3e}"
        )
    
    total_loss = (wIC * mse_ic + wBC_bottom * mse_bottom + wBC_left * mse_left 
                  + wBC_right * mse_right + wBC_top_heat * loss_heat + wBC_top_cold * loss_conv 
                  + wBC_top_heat * loss_pin_side + wMelt * melt_penalty + wPDE_bulk * loss_pde_bulk + wPDE_pin * loss_pde_pin)
    
    if torch.isnan(total_loss):
        return torch.tensor(1e6, device=device, requires_grad=True)
        
    return total_loss

# =================================================================
# In progress visualizer
# =================================================================

nx_viz, ny_viz = 80, 40
xv = np.linspace(x_min, x_max, nx_viz)
yv = np.linspace(y_min, y_max, ny_viz)
Xv, Yv = np.meshgrid(xv, yv, indexing="ij")

xv_t = torch.tensor(Xv.reshape(-1,1), device=device)
yv_t = torch.tensor(Yv.reshape(-1,1), device=device)

t_viz = torch.full_like(xv_t, t_max)   # final time
rpm_viz = torch.full_like(xv_t, rpm_ref)

def plot_temperature_field(net, iteration, save_dir):
    net.eval()
    with torch.no_grad():
        theta = net(xv_t, yv_t, t_viz, rpm_viz)
        theta = theta.cpu().numpy().reshape(nx_viz, ny_viz)

    # Physical temperature
    DeltaT_r = q_ref * L / k
    T_plot = T_room + DeltaT_r * theta

    fig, ax = plt.subplots(figsize=(6,4))
    cf = ax.contourf(
        Xv * L,
        Yv * L,
        T_plot,
        50,
        cmap="inferno"
    )
    fig.colorbar(cf, ax=ax, label="Temperature (°C)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Iteration {iteration}")
    ax.set_aspect("equal")
    ax.plot(shoulder_right_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    ax.plot(shoulder_left_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    rect = patches.Rectangle(
        (pin_left, pin_bottom), 
        pin_width, 
        Hp_raw, 
        linewidth=1.5, 
        edgecolor='cyan',
        facecolor='none', 
        linestyle='--'
    )
    ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(save_dir / f"temp_iter_{iteration:05d}.png", dpi=200)
    plt.close(fig)

    net.train()

# ==================================================================
# Training Loop
# ==================================================================

# empty array for storage
sanity_history = []

# Adam

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_history_adam = []

anchor_rpms = torch.tensor([230.0, 310.0, 410.0], device=device)

for i in range(1000): 
    optimizer.zero_grad()
    
    batch_loss = 0
    for r in anchor_rpms:
        batch_loss += pinn_loss(r.view(1), i)
    
    loss = batch_loss / 3
    loss.backward()

    loss_history_adam.append(loss.item())
    
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    if i % 100 == 0:
        net.eval()
        with torch.no_grad():
            theta_v = net(xv_t, yv_t, t_viz, rpm_viz).cpu().numpy().reshape(nx_viz, ny_viz)
            theta_min, theta_max, theta_mean = theta_v.min(), theta_v.max(), theta_v.mean()
            T_v = T_room + DeltaT_r * theta_v
            top_avg = T_v[:, -1].mean()
            bottom_avg = T_v[:, 0].mean()
            sanity_history.append((i, theta_min, theta_max, theta_mean, top_avg, bottom_avg))
            print(f"Sanity Iter {i}: θ_min={theta_min:.3e} θ_max={theta_max:.3e} θ_mean={theta_mean:.3e} top_avg={top_avg:.2f}C bottom_avg={bottom_avg:.2f}C")
        
            # Check temperature at the heating point (center x, top y)
            x_heat = torch.full((1,1), (x_min + x_max)/2, device=device)
            y_heat = torch.full((1,1), y_max, device=device)
            t_heat = torch.full((1,1), t_max, device=device)
            rpm_heat = torch.full((1,1), 410.0, device=device)
    
            theta_heat = net(x_heat, y_heat, t_heat, rpm_heat).item()
            T_heat = T_room + (q_ref * L / k) * theta_heat  
    
            print(f"At heating center: θ = {theta_heat:.4f}, T = {T_heat:.2f}°C")
            print(f"Expected: T ≈ {exp_T_at_ref:.2f}°C")
        net.train()

    if i % 200 == 0:
        print(f"Iter {i}: loss = {loss.item():.3e}")
        plot_temperature_field(net, i, results_dir)

print("Adam training complete.")

# LBFGS

rpm_lbfgs = torch.tensor(
    [200.0, 300.0, 410.0],
    device=device
)

loss_history_lbfgs = []

closure_iter = [0]  # counter

def closure():
    optimizer_lbfgs.zero_grad()
    total_loss = 0.0
    
    for r in rpm_lbfgs:
        loss = pinn_loss(r.view(1), iteration=2000) 
        total_loss += loss

    avg_loss = total_loss / 3

    closure_iter[0] += 1
    if closure_iter[0] % 20 == 0: # Print every 20 function evaluations
        print(f"LBFGS Eval {closure_iter[0]}: Loss = {avg_loss.item():.6e}")
        
    avg_loss.backward()
    
    # Track history for plotting
    loss_history_lbfgs.append(avg_loss.item())
    return avg_loss

optimizer_lbfgs = torch.optim.LBFGS(
    net.parameters(),
    lr=0.5, 
    max_iter=500,
    line_search_fn="strong_wolfe"
)

print("Starting LBFGS refinement...")
optimizer_lbfgs.step(closure)
print("Final LBFGS loss:", closure().item())
print("LBFGS training complete.")

# ==================================================================
# Scaling Back
# ==================================================================

nx, ny, nt = 200, 100, 60
xg = np.linspace(x_min,x_max,nx)
yg = np.linspace(y_min,y_max,ny)
tg = np.linspace(t_min,t_max,nt)

Xg,Yg,Tg = np.meshgrid(xg,yg,tg,indexing="ij")

with torch.no_grad():
    x_input = torch.tensor(Xg.reshape(-1,1), device=device)
    y_input = torch.tensor(Yg.reshape(-1,1), device=device)
    t_input = torch.tensor(Tg.reshape(-1,1), device=device)
    rpm_input = torch.full_like(x_input, rpm_ref) 
    
    theta_pred = net(x_input, y_input, t_input, rpm_input).cpu().numpy().reshape(Xg.shape)
    print("theta_pred range:", theta_pred.min(), theta_pred.max())

DeltaT_r = q_ref * L / k

flux_ratio_ref = 1.0  # at 410 RPM
T_pred = T_room + flux_ratio_ref * DeltaT_r * theta_pred

# For knowing what's going on
print("q_ref:", q_ref)
print("DeltaT_r:", DeltaT_r)
print("Top avg T:", np.mean(T_pred[:, -1, -1]))
print("Bottom avg T:", np.mean(T_pred[:, 0, -1]))
print("theta_pred range:", theta_pred.min(), theta_pred.max())
print("T_pred range:", T_pred.min(), T_pred.max())

# =================================================================
# Plots and stats
# =================================================================

# temp at final time
fig, ax = plt.subplots(figsize=(6,2))
plt.contourf(
    Xg[:, :, -1] * L,
    Yg[:, :, -1] * L,
    T_pred[:, :, -1],
    50,
    cmap="inferno"
)

ax.plot(shoulder_right_x, shoulder_y, color='white', linestyle='--', linewidth=2)
ax.plot(shoulder_left_x, shoulder_y, color='white', linestyle='--', linewidth=2)
rect = patches.Rectangle(
    (pin_left, pin_bottom), 
    pin_width, 
    Hp_raw, 
    linewidth=1.5, 
    edgecolor='cyan',
    facecolor='none', 
    linestyle='--'
)
ax.add_patch(rect)

ax.set_aspect("equal") 
plt.colorbar(label="Temperature (°C)")
plt.xlabel("x (m)")
plt.ylabel(f"y (m)")
plt.title("Temperature at final time")
plt.tight_layout()
fig.savefig(results_dir / "temperature_final.png", dpi=300)
plt.close(fig)


# GIF (temp over time)
fig, ax = plt.subplots(figsize=(6,5))
levels = np.linspace(T_pred.min(), T_pred.max(), 50)

def update(frame):
    ax.clear()
    ax.contourf(
        Xg[:,:,frame].T * L,
        Yg[:,:,frame].T * L,
        T_pred[:,:,frame].T,
        levels=levels,
        cmap="inferno"
    )
    ax.set_title(f"t = {tg[frame]*t_c:.2f} s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

ani = FuncAnimation(fig, update, frames=nt, interval=100)
gif_path = results_dir / "temperature_evolution.gif"
ani.save(gif_path, writer=PillowWriter(fps=10))
plt.close(fig)

print(f"Saved GIF: {gif_path}")

# Average temperature vs time
x_mid = nx // 2

avg_temp = np.mean(T_pred[x_mid, :, :], axis=0)

time_phys = tg * t_c

fig = plt.figure(figsize=(6,4))
plt.plot(time_phys, avg_temp, lw=2)
plt.xlabel("Time (s)")
plt.ylabel("Average temperature (°C)")
plt.title("Average temperature over domain")
plt.grid(True)
plt.tight_layout()
fig.savefig(results_dir / "average_temperature_vs_time.png", dpi=300)
plt.close(fig)

np.savetxt(
    results_dir / "average_temperature_vs_time.txt",
    np.column_stack((time_phys, avg_temp)),
    header="time(s) avg_temperature(C)",
    fmt="%.6e"
)

# Temp gradient (∂T/∂y)
dTdy = np.gradient(T_pred[:, :, -1], yg * L, axis=1)

fig, ax = plt.subplots(figsize=(6,5))

c = ax.contourf(
    Xg[:, :, -1].T * L,
    Yg[:, :, -1].T * L,
    dTdy.T,
    50,
    cmap="coolwarm"
)

fig.colorbar(c, ax=ax, label="∂T/∂y (°C/m)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Vertical temperature gradient")

ax.plot(shoulder_right_x, shoulder_y, color='white', linestyle='--', linewidth=2)
ax.plot(shoulder_left_x, shoulder_y, color='white', linestyle='--', linewidth=2)
rect = patches.Rectangle(
    (pin_left, pin_bottom), 
    pin_width, 
    Hp_raw, 
    linewidth=1.5, 
    edgecolor='cyan',
    facecolor='none', 
    linestyle='--'
)
ax.add_patch(rect)

fig.tight_layout()

fig.savefig(results_dir / "dTdy_final.png", dpi=300)
plt.close(fig)

# Different RPM

net.eval()

rpm_range = np.linspace(200, 450, 30)
final_avg_temps = []

# Spatial grid for averaging
Nx, Ny = 51, 31 # Ensure we have a point at the center
x_vals = np.linspace(x_min, x_max, Nx)
y_vals = np.linspace(y_min, y_max, Ny)
Xflat, Yflat = np.meshgrid(x_vals, y_vals)

x_flat = torch.tensor(Xflat.flatten()[:,None], device=device)
y_flat = torch.tensor(Yflat.flatten()[:,None], device=device)
t_flat = torch.full_like(x_flat, t_max)   # final time

with torch.no_grad():
    for r in rpm_range:
        rpm_tensor = torch.full_like(x_flat, r)
        theta_pred = net(x_flat, y_flat, t_flat, rpm_tensor)
        temp_phys = T_room + DeltaT_r * theta_pred 
        
        mask_line = (torch.abs(x_flat - x_center) < 1e-6).squeeze()
        
        line_avg = temp_phys[mask_line].mean().item()
        final_avg_temps.append(line_avg)

final_avg_temps = np.array(final_avg_temps)

exp_rpm = [410, 360, 310, 410, 360, 310, 330, 280, 230]
exp_temp = [543, 454, 347, 507, 429, 327, 447, 367, 273]

plt.figure(figsize=(8, 5))
plt.plot(rpm_range, final_avg_temps, 'b-', label='PINN Prediction')
plt.scatter(exp_rpm, exp_temp, color='red', marker='x', label='Experimental Data')
plt.xlabel("RPM")
plt.ylabel("Average Final Temperature (°C)")
plt.title("PINN vs. Experimental Data")

plt.legend()
plt.grid(True, alpha=0.3)

fig = plt.gcf() 
fig.savefig(results_dir / "comparison.png", dpi=300)
plt.show()

# Avg temp final time
avg_temp_final = avg_temp[-1]

csv_path = Path("results") / "all_runs_summary.csv"

# Create header if file does not exist
write_header = not csv_path.exists()

with open(csv_path, "a") as f:
    if write_header:
        f.write(
            "timestamp,rpm,q_in_Wm2,avg_temp_final_C\n"
        )

    f.write(
        f"{timestamp},"
        f"{rpm_ref},"
        f"{q_ref:.6e},"
        f"{avg_temp_final:.3f}\n"
    )

print(f"Appended run data to: {csv_path}")

# Convergence stats and convergence plots
np.savetxt(
    results_dir / "loss_history_adam.txt",
    np.array(loss_history_adam),
    header="Adam loss",
    fmt="%.6e"
)

np.savetxt(
    results_dir / "loss_history_lbfgs.txt",
    np.array(loss_history_lbfgs),
    header="LBFGS loss (all evaluations)",
    fmt="%.6e"
)

# Adam
fig = plt.figure(figsize=(6,4))
plt.semilogy(loss_history_adam, lw=2)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Adam training convergence")
plt.grid(True, which="both")
plt.tight_layout()
fig.savefig(results_dir / "loss_convergence_adam.png", dpi=300)
plt.close(fig)

# LBFGS
fig = plt.figure(figsize=(6,4))
plt.semilogy(loss_history_lbfgs, lw=1)
plt.xlabel("Function evaluation")
plt.ylabel("Loss")
plt.title("LBFGS convergence")
plt.grid(True, which="both")
plt.tight_layout()
fig.savefig(results_dir / "loss_convergence_lbfgs.png", dpi=300)
plt.close(fig)

# Combined plot
fig = plt.figure(figsize=(6,4))

plt.semilogy(
    range(len(loss_history_adam)),
    loss_history_adam,
    label="Adam",
    lw=2
)

plt.semilogy(
    range(len(loss_history_adam),
          len(loss_history_adam) + len(loss_history_lbfgs)),
    loss_history_lbfgs,
    label="LBFGS",
    lw=1
)

plt.xlabel("Iteration / evaluation")
plt.ylabel("Loss")
plt.title("PINN training convergence")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()

fig.savefig(results_dir / "loss_convergence_combined.png", dpi=300)
plt.close(fig)

# Temp gradient (∂T/∂x)
dTdx = np.gradient(T_pred[:, :, -1], xg * L, axis=0)

fig, ax = plt.subplots(figsize=(6,5))

c = ax.contourf(
    Xg[:, :, -1].T * L,
    Yg[:, :, -1].T * L,
    dTdx.T,
    50,
    cmap="coolwarm"
)

fig.colorbar(c, ax=ax, label="∂T/∂x (°C/m)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Horizontal temperature gradient")

ax.plot(shoulder_right_x, shoulder_y, color='white', linestyle='--', linewidth=2)
ax.plot(shoulder_left_x, shoulder_y, color='white', linestyle='--', linewidth=2)
rect = patches.Rectangle(
    (pin_left, pin_bottom), 
    pin_width, 
    Hp_raw, 
    linewidth=1.5, 
    edgecolor='cyan',
    facecolor='none', 
    linestyle='--'
)
ax.add_patch(rect)

fig.tight_layout()

fig.savefig(results_dir / "dTdx_final.png", dpi=300)
plt.close(fig)
