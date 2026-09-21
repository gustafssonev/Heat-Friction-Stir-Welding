#  PINN Model for FSW using the 2D Heat Equation
# - Heat input from shoulder within diameter
# - Heat input from pin modeled as internal source term
# - Convective side boundaries and top boundary outside of shoulder diameter
# - Insulated bottom boundary
# - Workpiece initially at room temperature
# - Tool moving in z-direction, modeled through moving coordinate (xsi = z - v*t) and Peclet number

# %%
# Imports

import torch                                                  # PyTorch
import torch.nn as nn                                         # PyTorch's module for neural networks
import numpy as np                                            # NumPy, used for math and arrays
import matplotlib.pyplot as plt                               # Create graphs and plots
from pathlib import Path                                      # Handle file paths
from datetime import datetime                                 # Time, for results saving
from matplotlib.animation import FuncAnimation                # Simulations
from matplotlib.animation import PillowWriter                 # Saves as GIF
import matplotlib.patches as patches                          # Draw shapes on plots

# %%
# Device Settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

# %%
# results saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results") / f"pinn_scaled_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)
print(f"Results saved to: {results_dir}")

# %%
# Physical values/constants
# - Values for Al 2219
# - Values from matweb and Dr.S

# physical values
alpha = 4.564e-5        # thermal diffusivity (m²/s) for aluminum
k = 112.0               # thermal conductivity (W/m·K) for aluminum
T_room = 20.0           # room temperature (°C)
T_melt = 660.0          # melting point of aluminum (°C)
T_steel = 40.0          # temperature of steel backing (°C)
'''h_air is 5-25, I am not sure what value to use'''
h_air = 25              # convection coefficient for air (W/m²·K)
'''h_steel is 500-3000, I am not sure what value to use'''
h_steel = 1000          # convection coefficient for steel (W/m²·K)
''' check rho and cp values'''
rho = 2700.0            # density (kg/m³) for aluminum
cp = 900.0              # specific heat capacity (J/kg·K) for aluminum
eps = 1e-8              # epsilon (small value)
v_z = 0.0033            # transverse speed (m/s) (200 mm/min)

# %%
# domain
# values from experiment
a1, b1 = 0.0, 1.04775e-1      # x-domain (m)
a2, b2 = 0.0, 6.35e-3         # y-domain (m)
a3, b3 = 0.0, 0.5334          # z-domain (m)
t_end = 210.0                 # end time (s)
Rsh_raw = 0.01905             # shoulder radius (m)
Rp_raw = 0.00635              # pin radius (m)
Hp_raw = 0.0047752            # pin height (m)
x_center_raw = (a1 + b1) / 2  # center x coordinate
shoulder_left_x = [(x_center_raw - Rsh_raw), (x_center_raw - Rsh_raw)]  # shoulder left boundary (x coordinates)
shoulder_right_x = [(x_center_raw + Rsh_raw), (x_center_raw + Rsh_raw)] # shoulder right boundary (x coordinates)
shoulder_y = [a2, b2]                                                   # shoulder y range
pin_width = 2 * Rp_raw              # pin width for visualization
pin_left = x_center_raw - Rp_raw    # pin left boundary (x coordinate)
pin_right = x_center_raw + Rp_raw   # pin right boundary (x coordinate)
pin_bottom = b2 - Hp_raw            # pin bottom boundary (y coordinate)
# %%
# Dimensionless Domain
# - To simplify training

L = b2 - a2                     # length (m)
t_c = L**2 / alpha              # time scale (s)
x_min, x_max = a1 / L, b1 / L   # scaled x domain
y_min, y_max = a2 / L, b2 / L   # scaled y domain
z_min, z_max = a3 / L, b3 / L   # scaled z domain
t_min, t_max = 0.0, t_end / t_c # scaled time domain
coef_air = h_air * L / k        # biot number for convection BC
coef_steel = h_steel * L / k    # biot number for bottom
Rsh = Rsh_raw / L               # scaled shoulder radius
Rp = Rp_raw / L                 # scaled pin radius
Hp = Hp_raw / L                 # scaled pin height
x_center = (x_min + x_max) / 2  # center x-coordinate
Pe_z = v_z * L / alpha          # Péclet number in z-direction

# %%
# Experimental data

# T vs RPM data (from Dr.S)
exp_rpm = np.array([410, 360, 310, 410, 360, 310, 330, 280, 230], dtype=float)
exp_temp = np.array([543, 454, 347, 507, 429, 327, 447, 367, 273], dtype=float)

theta_steel = (T_steel - T_room)/(max(exp_temp) - T_room) # dimensionless temperature of steel backing

# %%
# Flux and temperature reference

# For heat input scaling
deltaT_exp = exp_temp - T_room           # expected temperature rise above room temp
q_exp = k * deltaT_exp / L               # expected heat flux

# Input rpm we want to test, set as a fixed parameter here
RPM = 410.0                                           # set the RPM to use for this run; change later if needed
M = 40.0                                              # torque (N·m), check value
omega = RPM * 2 * np.pi / 60.0                        # angular velocity (rad/s)

exp_T_at_ref = np.mean(exp_temp[exp_rpm == RPM])  # corresponding experimental temperature at that rpm
q_ref = k * (exp_T_at_ref - T_room) / L               # reference heat flux
DeltaT_r = q_ref * L / k                              # reference temperature rise

# Scaling flux
flux_exp_dimless = q_exp / q_ref                                          # dimensionless flux
a_flux_dimless, b_flux_dimless = np.polyfit(exp_rpm, flux_exp_dimless, 1) # linear fit
a_flux_dimless = float(a_flux_dimless)                                    # float for PyTorch compatibility
b_flux_dimless = float(b_flux_dimless)                                    # float for PyTorch compatibility

# %%
# Getters

# get radius of rotation
def get_radius(x):
    r = x - x_center
    return r

# get velocity of rotation
def get_velocity(rpm_tensor, radius):
    v = radius * (rpm_tensor * 2 * np.pi) / 60.0
    return v

# get Q: internal heat from pin (dimensionless)
def get_internal_source(x, y, z, t, rpm_tensor):
    r_local = torch.abs(get_radius(x))               # radial distance from center
    k_s = 200.0                                      # controls how sharp the edge of the pin is 
    mask_x = torch.sigmoid(k_s * (Rp - r_local))     # 1 inside pin radius, 0 outside, with smooth transition
    mask_y = torch.sigmoid(k_s * (y - (y_max - Hp))) # 1 above pin height, 0 below, with smooth transition
    combined_mask = mask_x * mask_y                  # 1 in pin region, 0 elsewhere, with smooth edges

    z_tool = Pe_z * t
    sigma_z = Rp
    z_profile = torch.exp(-((z - z_tool)**2) / (2 * sigma_z**2))

    q_magnitude = (a_flux_dimless * rpm_tensor + b_flux_dimless)
    P_ref = M * omega
    P_total = P_ref * q_magnitude

    V_pin = np.pi * Rp_raw**2 * Hp_raw
    q_raw = P_total / V_pin # W/m^3
    # Q = (t_end * q_raw) / (rho * cp * DeltaT_r) * combined_mask * z_profile # dimensionless internal heat source
    Q = (L**2 * q_raw) / (alpha * rho * cp * DeltaT_r) * combined_mask * z_profile # dimensionless internal heat source
    return Q

# %%
# Output for debugging purposes
print(f"q_ref (W/m²): {q_ref:.2f}")                                                     # output reference heat flux
print(f"DeltaT_r (°C): {DeltaT_r:.2f}")                                             # output reference temperature rise
print(f"Flux ratio coefficients: a={a_flux_dimless:.6f}, b={b_flux_dimless:.6f}")       # output linear fit coefficients
print(f"flux_ratio(reference RPM) = {a_flux_dimless * RPM + b_flux_dimless:.6f}")   # output flux ratio at reference RPM
print(f"Expected T_max at reference RPM: {T_room + DeltaT_r:.2f}°C")                  # output expected max temperature at reference RPM

# %%
# Collocation Points

Nf, Ni, Nb = 8000, 500, 3000 # number of collocation points for PDE, initial condition, and boundary conditions

# pde
x_f = torch.rand(Nf,1,device=device)*(x_max-x_min)+x_min
y_f = torch.rand(Nf,1,device=device)*(y_max-y_min)+y_min
z_f = torch.rand(Nf,1,device=device)*(z_max-z_min)+z_min
t_f = torch.rand(Nf,1,device=device)*(t_max-t_min)+t_min
x_f.requires_grad_(True)
y_f.requires_grad_(True)
z_f.requires_grad_(True)
t_f.requires_grad_(True)

# initial
x_i = torch.rand(Ni,1,device=device)*(x_max-x_min)+x_min
y_i = torch.rand(Ni,1,device=device)*(y_max-y_min)+y_min
z_i = torch.rand(Ni,1,device=device)*(z_max-z_min)+z_min
t_i = torch.zeros(Ni,1,device=device, requires_grad=True)

# bottom
x_b = torch.rand(Nb,1,device=device)*(x_max-x_min)+x_min
y_b = torch.full((Nb,1), y_min, device=device)
z_b = torch.rand(Nb,1,device=device)*(z_max-z_min)+z_min
t_b = torch.rand(Nb,1,device=device)*(t_max-t_min)+t_min
x_b.requires_grad_(True)
y_b.requires_grad_(True)

# top
Nb_tool = Nb // 2
Nb_rand = Nb - Nb_tool
x_t_rand = torch.rand(Nb_rand, 1, device=device) * (x_max - x_min) + x_min
x_t_tool = torch.rand(Nb_tool, 1, device=device) * (2 * Rsh) + (x_center - Rsh)

t_tool_max = min(t_max, z_max / Pe_z)

t_t_rand = torch.rand(Nb_rand, 1, device=device) * (t_max - t_min) + t_min
z_t_rand = torch.rand(Nb_rand, 1, device=device) * (z_max - z_min) + z_min

t_t_tool = torch.rand(Nb_tool, 1, device=device) * (t_tool_max - t_min) + t_min
z_t_tool = Pe_z * t_t_tool

x_t = torch.cat([x_t_rand, x_t_tool], dim=0)
y_t = torch.full((Nb,1), y_max, device=device)
z_t = torch.cat([z_t_rand, z_t_tool], dim=0)
t_t = torch.cat([t_t_rand, t_t_tool], dim=0)
x_t.requires_grad_(True)
y_t.requires_grad_(True)

# left
x_l = torch.full((Nb,1), x_min, device=device)
y_l = torch.rand(Nb,1,device=device)*(y_max-y_min)+y_min
z_l = torch.rand(Nb,1,device=device)*(z_max-z_min)+z_min
t_l = torch.rand(Nb,1,device=device)*(t_max-t_min)+t_min
x_l.requires_grad_(True)
y_l.requires_grad_(True)

# right (use same y, z, t as left for symmetry)
x_r = torch.full((Nb,1), x_max, device=device)
y_r = y_l.clone().detach().requires_grad_(True)  # Same as left
z_r = z_l.clone().detach().requires_grad_(True)  # Same as left
t_r = t_l.clone().detach().requires_grad_(True)  # Same as left
x_r.requires_grad_(True)
y_r.requires_grad_(True)

# %%
# Network
# - 8 hidden layers with 64 neurons: chosen after testing and comparing combinations of [32, 64, 128] and [4, 6, 8] (see "Short test block for tuning")
# - SiLU is used as nonlinear activation function
# - Softplus to ensure nonnegativity

num_neurons = 64
num_layers = 4
input_dim = 5
output_dim = 1

class PINN(nn.Module):
    def __init__(self, num_neurons, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, num_neurons), nn.SiLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), nn.SiLU()]
        layers.append(nn.Linear(num_neurons, output_dim))
        self.net = nn.Sequential(*layers)

        # Initialize weights to avoid 0 at start
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.out_features == output_dim:
                    nn.init.constant_(m.bias, 0.5) # output layer
                else:
                    nn.init.constant_(m.bias, 0.1) # hidden layers

        # Learnable weigths
        self.log_weights = nn.ParameterDict({
            'ic': nn.Parameter(torch.zeros(1)),
            'pde': nn.Parameter(torch.zeros(1)),
            'left': nn.Parameter(torch.zeros(1)),
            'right': nn.Parameter(torch.zeros(1)),
            'bottom': nn.Parameter(torch.zeros(1)),
            'top': nn.Parameter(torch.zeros(1))
        })

    def forward(self, x, y, z, t, rpm):
        x_half_width = (x_max - x_min) / 2
        x_norm = (x - x_center) / x_half_width
        y_norm = y / y_max
        z_norm = z / z_max
        t_norm = t / t_max
        rpm_norm = (rpm - 230.0) / (410.0 - 230.0)

        out = self.net(torch.cat([x_norm, y_norm, z_norm, t_norm, rpm_norm], dim=1))
        base_theta = torch.nn.functional.softplus(out) # softplus for > 0 output
        flux_ratio = a_flux_dimless * rpm + b_flux_dimless
        theta = base_theta * flux_ratio
        return theta

# Initialize network
net = PINN(num_neurons=64, num_layers=8).to(device)

# %%
# Loss Function
# - PDE: 2D heat equation: $(\theta_t + v_{x,scaled} * \theta_x) = (\theta_{xx} + \theta_{yy}) - Q_{internal,scaled}$ ($v_{x,scaled}=0$ currently)
# - Initial condition: $\theta(x,y,0) = 0$
# - Bottom BC: $\frac{\partial\theta}{\partial y} = 0$
# - Left BC: $\frac{\partial\theta}{\partial x} = \frac{hL\theta}{k}$
# - Right BC: $\frac{\partial\theta}{\partial x} = -\frac{hL\theta}{k}$
# - Top BC within tool diamater: RPM-scaled heat flux
# - Top BC outside tool diamater: $\frac{\partial\theta}{\partial y} = \frac{hL\theta}{k}$
# - Pin internal heat source through $Q_{internal, scaled}$ in heat equation

def pinn_loss(rpm_tensor=None, iteration=0, *, x_f, y_f, z_f, t_f, bc_top_multiplier=1.0):
    if rpm_tensor is None:
        rpm_tensor = torch.tensor([RPM], device=device)

    if rpm_tensor.dim() == 1:
        rpm_tensor = rpm_tensor.unsqueeze(1)

    # rpm_f = rpm_tensor.repeat(Nf, 1)
    rpm_f = rpm_tensor.repeat(x_f.shape[0], 1)

    theta = net(x_f, y_f, z_f, t_f, rpm_f)

    # derivatives

    theta_x = torch.autograd.grad(
        theta, x_f, torch.ones_like(theta), create_graph=True, retain_graph=True
    )[0]

    theta_y = torch.autograd.grad(
        theta, y_f, torch.ones_like(theta), create_graph=True, retain_graph=True
    )[0]

    theta_z = torch.autograd.grad(
        theta, z_f, torch.ones_like(theta), create_graph=True, retain_graph=True
    )[0]

    theta_t = torch.autograd.grad(
        theta, t_f, torch.ones_like(theta), create_graph=True, retain_graph=True
    )[0]

    theta_xx = torch.autograd.grad(
        theta_x, x_f, torch.ones_like(theta_x), create_graph=False, retain_graph=True
    )[0]

    theta_yy = torch.autograd.grad(
        theta_y, y_f, torch.ones_like(theta_y), create_graph=False, retain_graph=True
    )[0]

    theta_zz = torch.autograd.grad(
        theta_z, z_f, torch.ones_like(theta_z), create_graph=False, retain_graph=True
    )[0]

    # mask
    # --------------------------------------------------------------------------

    sharpness = 10.0 # Adjustable sharpness for the mask (trying 5 instead of 50)
    
    # heat equation
    # Since get_internal_source already has mask, we don't need to apply an additional mask here
    # --------------------------------------------------------------------------

    Q_internal = get_internal_source(x_f, y_f, z_f, t_f, rpm_f)

    # residual = theta_t - (alpha * t_end / L**2 ) * (theta_xx + theta_yy + theta_zz) - Q_internal
    residual = theta_t - (theta_xx + theta_yy + theta_zz) - Q_internal
    mse_pde = torch.mean(residual**2)

    # Initial condition: θ(x,y,0) = 0
    # --------------------------------------------------------------------------

    theta_ic = net(x_i, y_i, z_i, t_i, rpm_tensor.repeat(Ni, 1))
    mse_ic = torch.mean(theta_ic ** 2)

    # Bottom BC: ∂θ/∂y = conv
    # --------------------------------------------------------------------------

    theta_bottom = net(x_b, y_b, z_b, t_b, rpm_tensor.repeat(Nb, 1))

    theta_yb = torch.autograd.grad(
        theta_bottom, y_b,
        torch.ones_like(theta_bottom),
        create_graph=False, retain_graph=True
    )[0]

    # Bottom BC: heat flows out through steel backing
    # dθ/dy = -coef_steel * (theta - theta_steel)  (negative because +y points out of domain)
    # mse_bottom = torch.mean(theta_yb ** 2)  # insulated: dtheta/dy = 0
    mse_bottom = torch.mean((theta_yb - coef_steel * (theta_bottom - theta_steel))**2)

    # Left BC: ∂θ/∂x = hLθ/k
    # --------------------------------------------------------------------------

    theta_left = net(x_l, y_l, z_l, t_l, rpm_tensor.repeat(Nb, 1))

    theta_xl = torch.autograd.grad(
        theta_left, x_l,
        torch.ones_like(theta_left),
        create_graph=False, retain_graph=True
    )[0]

    mse_left = torch.mean((theta_xl - coef_air * theta_left) ** 2)

    # Right BC: ∂theta/∂x = -hLtheta/k
    # --------------------------------------------------------------------------

    theta_right = net(x_r, y_r, z_r, t_r, rpm_tensor.repeat(Nb, 1))

    theta_xr = torch.autograd.grad(
        theta_right, x_r,
        torch.ones_like(theta_right),
        create_graph=False, retain_graph=True
    )[0]

    mse_right = torch.mean((theta_xr + coef_air * theta_right) ** 2)

    # Top BC: RPM-scaled heat flux
    # --------------------------------------------------------------------------

    theta_top = net(x_t, y_t, z_t, t_t, rpm_tensor.repeat(Nb, 1))

    theta_yt = torch.autograd.grad(
        theta_top, y_t,
        torch.ones_like(theta_top),
        create_graph=False, retain_graph=True
    )[0]

    flux_ratio = (a_flux_dimless * rpm_tensor + b_flux_dimless) # dimensionless flux ratio at current RPM
    flux_target = flux_ratio.expand_as(theta_yt)                # target dimensionless flux at each collocation point on the top boundary

    # Compute mask on top boundary

    dist_from_tool_t = torch.sqrt((x_t - x_center)**2 + (z_t - Pe_z * t_t)**2 + 1e-8)
    mask_inner = torch.sigmoid(sharpness * (dist_from_tool_t - Rp))  # 1 outside Rp, 0 inside
    mask_outer = torch.sigmoid(sharpness * (Rsh - dist_from_tool_t))  # 1 inside Rsh, 0 outside
    mask_shoulder_t = mask_inner * mask_outer  # no heat from within pin radius

    # Top BC:
    # - Shoulder region (mask_shoulder_t): heat flux into domain -> dtheta/dy = -flux_target
    # - Outer region (1 - mask_shoulder_t): convection out -> dtheta/dy = +coef_air * theta
    
    w_extra_heat = 2.0  # weight for enforcing heat flux (compared to convection loss)
    
    # Heat flux BC: dθ/dy = -flux_target 
    mask_sum = torch.sum(mask_shoulder_t) + eps
    loss_heat = torch.sum(mask_shoulder_t * (theta_yt + flux_target)**2) / mask_sum
    
    # Convection BC: dθ/dy = +coef_air * theta
    conv_sum = torch.sum(1 - mask_shoulder_t) + eps
    loss_conv = torch.sum((1 - mask_shoulder_t) * (theta_yt - coef_air * theta_top)**2) / conv_sum

    mse_top = loss_heat * w_extra_heat + loss_conv

    # Weights and sum
    #---------------------------------------------------------------------------

    # Debugging prints
    if iteration % 100 == 0:
        rpm_heat_diag = torch.full((100,1), RPM, device=device)
        
        with torch.no_grad():
            # Check temperature at shoulder region (center)
            x_shoulder = torch.full((50,1), x_center, device=device)
            y_shoulder = torch.full((50,1), y_max, device=device)
            z_shoulder = torch.linspace(z_min, z_max, 50, device=device).view(-1,1)
            t_shoulder = torch.full((50,1), t_max, device=device)
            theta_shoulder = net(x_shoulder, y_shoulder, z_shoulder, t_shoulder, rpm_heat_diag[:50])
            
            # Check temperature at right edge (x_max)
            x_right = torch.full((50,1), x_max, device=device)
            theta_right = net(x_right, y_shoulder, z_shoulder, t_shoulder, rpm_heat_diag[:50])
            
            # Check temperature at left edge (x_min)
            x_left = torch.full((50,1), x_min, device=device)
            theta_left = net(x_left, y_shoulder, z_shoulder, t_shoulder, rpm_heat_diag[:50])
        
        print(
            f"loss_pde={mse_pde.item():.3e} "
            f"loss_heat={loss_heat.item():.3e} "
            f"loss_conv={loss_conv.item():.3e} "
            f"mask_mean={mask_shoulder_t.mean().item():.3f} "
            f"mask_count={mask_sum.item():.1f} "
            f"| θ@shoulder={theta_shoulder.mean().item():.4f} "
            f"θ@left={theta_left.mean().item():.4f} "
            f"θ@right={theta_right.mean().item():.4f}"
        )

    # Loss weights 
    wIC = 10.0  
    wBC_bottom = 50.0  
    wBC_left = 2.0  
    wBC_right = 2.0  
    wBC_top = 100.0  
    wPDE = 5.0  

    adapt_start = 300.0 # when we start learnable weights
    
    # make learnable
    log_ic = torch.exp(-net.log_weights['ic'])
    log_bottom = torch.exp(-net.log_weights['bottom'])
    log_left = torch.exp(-net.log_weights['left'])
    log_right = torch.exp(-net.log_weights['right'])
    log_pde = torch.exp(-net.log_weights['pde'])
    log_top = torch.exp(-net.log_weights['top'])

    # we need add term for learnable
    add_ic = 0 if iteration < adapt_start else log_ic
    add_bottom = 0 if iteration < adapt_start else log_bottom
    add_left = 0 if iteration < adapt_start else log_left
    add_right = 0 if iteration < adapt_start else log_right
    add_pde = 0 if iteration < adapt_start else log_pde
    add_top = 0 if iteration < adapt_start else log_top

    # which one to use
    total_ic = wIC if iteration < adapt_start else log_ic
    total_bottom = wBC_bottom if iteration < adapt_start else log_bottom
    total_left = wBC_left if iteration < adapt_start else log_left
    total_right = wBC_right if iteration < adapt_start else log_right
    total_pde = wPDE if iteration < adapt_start else log_pde
    total_top = wBC_top if iteration < adapt_start else log_top

    if torch.isnan(mse_pde).any() or torch.isinf(mse_pde).any():
        return torch.tensor(1e6, device=device, requires_grad=True)

    total_loss = (total_ic * mse_ic + add_ic
                + total_bottom * mse_bottom + add_bottom
                + total_left * mse_left + add_left
                + total_right * mse_right + add_right
                + total_pde * mse_pde + add_pde
                + total_top * mse_top + add_top  
                 )
    
    if torch.isnan(total_loss).any():
        return torch.tensor(1e6, device=device, requires_grad=True)

    return total_loss

# %%
# Visualizer

nx_viz, ny_viz = 80, 40
xv = np.linspace(x_min, x_max, nx_viz)
yv = np.linspace(y_min, y_max, ny_viz)
Xv, Yv = np.meshgrid(xv, yv, indexing="ij")

xv_t = torch.tensor(Xv.reshape(-1,1), device=device)
yv_t = torch.tensor(Yv.reshape(-1,1), device=device)

z_viz = torch.full_like(xv_t, z_max/2)
t_viz = torch.full_like(xv_t, t_max)   # final time
rpm_viz = torch.full_like(xv_t, RPM)
xi_viz = torch.zeros_like(xv_t)

def get_tool_z_position(t_dimless):
    """Return the tool z position in dimensionless coordinates."""
    z_tool = Pe_z * t_dimless
    return min(max(z_tool, z_min), z_max)


def plot_temperature_field(net, iteration, save_dir, z_slice=None, t_slice=None):
    net.eval()

    if z_slice is None:
        if t_slice is None:
            z_slice = z_max / 2
        else:
            z_slice = get_tool_z_position(t_slice)

    if t_slice is None:
        t_slice = t_max

    z_viz_slice = torch.full_like(xv_t, z_slice)
    t_viz_slice = torch.full_like(xv_t, t_slice)

    with torch.no_grad():
        theta = net(xv_t, yv_t, z_viz_slice, t_viz_slice, rpm_viz)
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
    ax.set_title(f"Iteration {iteration} at z={z_slice*L:.4f} m")
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
    ax.scatter([x_center*L], [y_max*L], color='cyan', s=30, marker='x', label='tool x-center')
    ax.legend(loc='upper right', fontsize='small')

    fig.tight_layout()
    fig.savefig(save_dir / f"temp_iter_{iteration:05d}_zslice.png", dpi=200)
    plt.close(fig)

    net.train()

# %%
# This is a short testing block (for various tests)

def train_model_quick(hidden_size, num_layers):

    net = PINN(hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    anchor_rpms = torch.tensor([RPM], device=device)

    for i in range(300):
        optimizer.zero_grad()

        batch_loss = 0
        for r in anchor_rpms:
            batch_loss += pinn_loss(r.view(1), i, x_f=x_f, y_f=y_f, z_f=z_f, t_f=t_f)

        loss = batch_loss / 3
        loss.backward(retain_graph=True)
        optimizer.step()

    return net


# %%
# Adam training

sanity_history = []
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6
)
loss_history_adam = []
anchor_rpms = torch.tensor([RPM], device=device)

# Early stopping parameters
best_loss = float('inf')
patience = 500
counter = 0

for i in range(800):
    optimizer.zero_grad()

    # Resampling
    x_f = torch.rand(Nf, 1, device=device) * (x_max - x_min) + x_min
    y_f = torch.rand(Nf, 1, device=device) * (y_max - y_min) + y_min
    z_f = torch.rand(Nf, 1, device=device) * (z_max - z_min) + z_min
    t_f = torch.rand(Nf, 1, device=device) * (t_max - t_min) + t_min
    
    x_f.requires_grad_(True)
    y_f.requires_grad_(True)
    z_f.requires_grad_(True)
    t_f.requires_grad_(True)

    batch_loss = 0
    for r in anchor_rpms:
        batch_loss += pinn_loss(r.view(1), i, x_f=x_f, y_f=y_f, z_f=z_f, t_f=t_f)
    loss = batch_loss  
    loss.backward()
    
    # Gradient clipping to prevent instability
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    loss_history_adam.append(loss.item())

    optimizer.step()
    
    # Step the scheduler
    scheduler.step(loss)

    # Early stopping check
    if loss.item() < best_loss - 1e-7:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Adam converged and stopped early at iteration {i}")
        break

    if i % 100 == 0:
        net.eval()
        with torch.no_grad():
            theta_v = net(xv_t, yv_t, z_viz, t_viz, rpm_viz).cpu().numpy().reshape(nx_viz, ny_viz)
            theta_min, theta_max, theta_mean = theta_v.min(), theta_v.max(), theta_v.mean()
            T_v = T_room + DeltaT_r * theta_v
            top_avg = T_v[:, -1].mean()
            bottom_avg = T_v[:, 0].mean()
            sanity_history.append((i, theta_min, theta_max, theta_mean, top_avg, bottom_avg))
            print(f"Sanity Iter {i}: θ_min={theta_min:.3e} θ_max={theta_max:.3e} θ_mean={theta_mean:.3e} top_avg={top_avg:.2f}C bottom_avg={bottom_avg:.2f}C")

            # Check temperature at the heating point (center x, top y)
            x_heat = torch.full((1,1), (x_min + x_max)/2, device=device)
            y_heat = torch.full((1,1), y_max, device=device)
            z_heat = torch.full((1,1), z_max/2, device=device)
            t_heat = torch.full((1,1), t_max, device=device)
            rpm_heat = torch.full((1,1), RPM, device=device)

            theta_heat = net(x_heat, y_heat, z_heat, t_heat, rpm_heat).item()
            T_heat = T_room + (q_ref * L / k) * theta_heat

            print(f"At heating center: θ = {theta_heat:.4f}, T = {T_heat:.2f}°C")
            print(f"Expected: T ≈ {exp_T_at_ref:.2f}°C")
        net.train()

    if i % 200 == 0:
        print(f"Iter {i}: loss = {loss.item():.3e}")
        plot_temperature_field(net, i, results_dir)
        tool_time = min(t_max, z_max / Pe_z)
        plot_temperature_field(net, i, results_dir, t_slice=tool_time)

print("Adam training complete.")

# %%
# LBFGS Refinement

rpm_lbfgs = torch.tensor(
    [RPM],  
    device=device
)

loss_history_lbfgs = []
closure_iter = [0]

# Freeze adaptive weights from Adam
for name, param in net.log_weights.items():
    param.requires_grad = False

def closure():
    optimizer_lbfgs.zero_grad()
    total_loss = 0.0

    for r in rpm_lbfgs:
        loss = pinn_loss(r.view(1), 2000, x_f=x_f, y_f=y_f, z_f=z_f, t_f=t_f)
        total_loss += loss
    
    # Only backward once at the end
    total_loss.backward(retain_graph=True)
    
    closure_iter[0] += 1
    if closure_iter[0] % 10 == 0:
        print(f"LBFGS Eval {closure_iter[0]}: Loss = {total_loss.item():.6e}")
    
    loss_history_lbfgs.append(total_loss.item())
    return total_loss

optimizer_lbfgs = torch.optim.LBFGS(
    net.parameters(),
    lr=0.01,  # Much lower learning rate to prevent instability
    max_iter=20,  # Fewer iterations to prevent blowup
    line_search_fn="strong_wolfe"  # More careful line search
)

print("Starting LBFGS refinement...")
try:
    optimizer_lbfgs.step(closure)
except Exception as e:
    print(f"LBFGS stopped: {e}")
print(f"LBFGS training complete. Final loss: {loss_history_lbfgs[-1]:.6e}" if loss_history_lbfgs else "No iterations completed.")

# %%
# Scaling back

nx, ny, nz, nt = 50, 50, 50, 50
xg = np.linspace(x_min,x_max,nx)
yg = np.linspace(y_min,y_max,ny)
zg = np.linspace(z_min,z_max,nz)
tg = np.linspace(t_min,t_max,nt)

Xg,Yg,Zg,Tg = np.meshgrid(xg,yg,zg,tg,indexing="ij")

with torch.no_grad():
    x_input = torch.tensor(Xg.reshape(-1,1), device=device)
    y_input = torch.tensor(Yg.reshape(-1,1), device=device)
    z_input = torch.tensor(Zg.reshape(-1,1), device=device)
    t_input = torch.tensor(Tg.reshape(-1,1), device=device)
    rpm_input = torch.full_like(x_input, RPM)


    theta_pred = net(x_input, y_input, z_input, t_input, rpm_input).cpu().numpy().reshape(Xg.shape)
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

print(f"Reference Flux (q_ref): {q_ref:.2e}")
print(f"Dimensionless Scaling Factor (L^2 / (k * DeltaT_r)): {(L**2 / (k * DeltaT_r)):.2e}")

# Test the internal source magnitude
test_rpm = torch.tensor([410.0], device=device)

# Temperature at final time graph


# %%
# Plots

# temp at final time
fig, ax = plt.subplots(figsize=(8,6))
plt.contourf(
    Xg[:, :, 0, -1] * L,
    Yg[:, :, 0, -1] * L,
    T_pred[:, :, 0, -1],
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
plt.show()

# Temperature GIF

# GIF (temp over time)
fig, ax = plt.subplots(figsize=(6,5))
levels_min = float(T_pred.min())
levels_max = float(T_pred.max())
if np.isclose(levels_min, levels_max):
    levels = np.array([levels_min - 1e-3, levels_min + 1e-3])
else:
    levels = np.linspace(levels_min, levels_max, 50)

def update(frame):
    ax.clear()
    ax.contourf(
        Xg[:,:,0,frame] * L,
        Yg[:,:,0,frame] * L,
        T_pred[:, :, 0, frame],
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
plt.show()

print(f"Saved GIF: {gif_path}")

# Average temperature vs time

# Average temperature vs time
x_mid = nx // 2

avg_temp = np.mean(T_pred[x_mid, :, :, :], axis=(1, 2))

time_phys = tg * t_c

fig = plt.figure(figsize=(6,4))
plt.plot(time_phys, avg_temp, lw=2)
plt.xlabel("Time (s)")
plt.ylabel("Average temperature (°C)")
plt.title("Average temperature over domain")
plt.grid(True)
plt.tight_layout()
fig.savefig(results_dir / "average_temperature_vs_time.png", dpi=300)
plt.show()

np.savetxt(
    results_dir / "average_temperature_vs_time.txt",
    np.column_stack((time_phys, avg_temp)),
    header="time(s) avg_temperature(C)",
    fmt="%.6e"
)

# Temperature gradient y direction

# Temp gradient (∂T/∂y)
dTdy = np.gradient(T_pred[:, :, 0, -1], yg * L, axis=1)

fig, ax = plt.subplots(figsize=(8,6))

c = ax.contourf(
    Xg[:, :, 0, -1] * L,
    Yg[:, :, 0, -1] * L,
    dTdy,
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
plt.show()

# Temperature gradient x direction

# Temp gradient (∂T/∂x)
dTdx = np.gradient(T_pred[:, :, 0, -1], xg * L, axis=0)

fig, ax = plt.subplots(figsize=(8,6))

c = ax.contourf(
    Xg[:, :, 0, -1] * L,
    Yg[:, :, 0, -1] * L,
    dTdx,
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
plt.show()

# Average temperature data

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
        f"{RPM},"
        f"{q_ref:.6e},"
        f"{avg_temp_final:.3f}\n"
    )

print(f"Appended run data to: {csv_path}")

# Convergence data

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

# more results visualization

xi_line = torch.linspace(-3*Rp, 3*Rp, 200).view(-1,1).to(device)
x_line = torch.zeros_like(xi_line)        # center
y_line = torch.zeros_like(xi_line)        # mid-depth
rpm_line = torch.full_like(xi_line, RPM)

with torch.no_grad():
    z_abs_line = xi_line + Pe_z * torch.zeros_like(xi_line)
    theta = net(x_line, y_line, z_abs_line, torch.zeros_like(xi_line), rpm_line).cpu().numpy()

plt.plot(xi_line.cpu().numpy(), theta)
plt.xlabel("ξ")
plt.ylabel("θ")
plt.title("Temperature profile along weld direction")
plt.grid()
plt.show()

xi_values = [-2*Rp, -Rp, 0.0, Rp, 2*Rp]

fig, axes = plt.subplots(1, len(xi_values), figsize=(15,3))

for i, xi_val in enumerate(xi_values):
    xi_viz = torch.full_like(xv_t, xi_val)

    with torch.no_grad():
        z_abs = xi_viz + Pe_z * torch.zeros_like(xv_t)
        theta = net(xv_t, yv_t, z_abs, torch.zeros_like(xv_t), rpm_viz)
        theta = theta.cpu().numpy().reshape(nx_viz, ny_viz)

    T_plot = T_room + (q_ref * L / k) * theta

    ax = axes[i]
    cf = ax.contourf(Xv * L, Yv * L, T_plot, 50, cmap="inferno")

    ax.set_title(f"ξ = {xi_val:.2f}")
    ax.set_aspect("equal")

plt.colorbar(cf, ax=axes, label="Temperature (°C)")
plt.tight_layout()
plt.show()

# y=y_max

# Visualization: Temperature on top surface (y=y_max) in x-xsi plane
nx_surf, nxi_surf = 80, 40  # Grid resolution (adjust as needed)
x_surf = np.linspace(x_min, x_max, nx_surf)
xi_surf = np.linspace(0.0, z_max, nxi_surf)  # xsi from start to end of domain
X_surf, Xi_surf = np.meshgrid(x_surf, xi_surf, indexing="ij")

x_surf_t = torch.tensor(X_surf.reshape(-1, 1), device=device)
xi_surf_t = torch.tensor(Xi_surf.reshape(-1, 1), device=device)
y_surf_t = torch.full_like(x_surf_t, y_max)  # Fixed at top surface
t_surf_t = torch.full_like(x_surf_t, t_max/2)  # Fixed at final time
rpm_surf_t = torch.full_like(x_surf_t, RPM)  # Fixed RPM

net.eval()
with torch.no_grad():
    z_abs_surf = xi_surf_t + Pe_z * t_surf_t
    theta_surf = net(x_surf_t, y_surf_t, z_abs_surf, t_surf_t, rpm_surf_t).cpu().numpy().reshape(nx_surf, nxi_surf)
    T_surf = T_room + DeltaT_r * theta_surf

fig, ax = plt.subplots(figsize=(8, 4))
cf = ax.contourf(X_surf * L, Xi_surf * L, T_surf, 50, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperature (°C)")
ax.set_xlabel("x (m)")
ax.set_ylabel("ξ (m)")  # xsi direction (weld path)
ax.set_title(f"Temperature on Top Surface at t = {t_max/2 * t_c:.2f} s, RPM = {RPM}")
ax.set_aspect("equal")
ax.axhline(y=0, color='white', linestyle='--', linewidth=1, label='Start of weld')
ax.axhline(y=z_max * L, color='white', linestyle='--', linewidth=1, label='End of weld')
ax.legend()
fig.tight_layout()
fig.savefig(results_dir / "top_surface_xsi_contour.png", dpi=300)
plt.show()
net.train()

# xy planes

# Visualization: xy planes at specific xsi values (cross-sections along weld)
xi_values = [0.0, z_max/2, z_max]  # Example xsi values (adjust to your domain, e.g., 0, z_max/2, z_max)

fig, axes = plt.subplots(1, len(xi_values), figsize=(15, 4))

for i, xi_val in enumerate(xi_values):
    xi_viz = torch.full_like(xv_t, xi_val)  # Fixed xsi
    t_viz = torch.full_like(xv_t, t_max)    # Fixed at final time
    rpm_viz = torch.full_like(xv_t, RPM)  # Fixed RPM

    net.eval()
    with torch.no_grad():
        z_abs = xi_viz + Pe_z * t_viz
        theta = net(xv_t, yv_t, z_abs, t_viz, rpm_viz)
        theta = theta.cpu().numpy().reshape(nx_viz, ny_viz)
        T_plot = T_room + DeltaT_r * theta
    
    ax = axes[i]
    cf = ax.contourf(Xv * L, Yv * L, T_plot, 50, cmap="inferno")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    # Add boundaries for context
    ax.plot(shoulder_right_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    ax.plot(shoulder_left_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    rect = patches.Rectangle((pin_left, pin_bottom), pin_width, Hp_raw, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)

fig.colorbar(cf, ax=axes, label="Temperature (°C)")
plt.tight_layout()
fig.savefig(results_dir / "xy_cross_sections_xsi.png", dpi=300)
plt.show()
net.train()

# Fixed z

# Visualization: xy planes at fixed physical z = b3/2 for t=0, t_max/2, t_max
z_fixed = b3 / 2  # Fixed physical z (m)
z_dim = z_fixed / L  # Dimensionless z
t_values = [0.0, t_max / 2, t_max]  # Dimensionless times

fig, axes = plt.subplots(1, len(t_values), figsize=(15, 4))

for i, t_val in enumerate(t_values):
    z_viz_fixed = torch.full_like(xv_t, z_dim)
    t_viz = torch.full_like(xv_t, t_val)
    rpm_viz = torch.full_like(xv_t, RPM)
    
    net.eval()
    with torch.no_grad():
        theta = net(xv_t, yv_t, z_viz_fixed, t_viz, rpm_viz)
        theta = theta.cpu().numpy().reshape(nx_viz, ny_viz)
        T_plot = T_room + DeltaT_r * theta
    
    ax = axes[i]
    cf = ax.contourf(Xv * L, Yv * L, T_plot, 50, cmap="inferno")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    # Add boundaries for context
    ax.plot(shoulder_right_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    ax.plot(shoulder_left_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    rect = patches.Rectangle((pin_left, pin_bottom), pin_width, Hp_raw, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)

fig.colorbar(cf, ax=axes, label="Temperature (°C)")
plt.tight_layout()
fig.savefig(results_dir / "xy_cross_sections_fixed_z_b3_half.png", dpi=300)
plt.show()
net.train()

# GIF over fixed z

z_fixed = b3 / 2
z_dim = z_fixed / L
t_frames = np.linspace(0, t_max, 50)  # 50 frames

fig, ax = plt.subplots(figsize=(6, 4))
levels = np.linspace(20, 700, 50)  # Adjust based on temp range

def update(frame):
    t_val = t_frames[frame]
    z_viz_fixed = torch.full_like(xv_t, z_dim)
    t_viz = torch.full_like(xv_t, t_val)
    rpm_viz = torch.full_like(xv_t, RPM)
    
    net.eval()
    with torch.no_grad():
        theta = net(xv_t, yv_t, z_viz_fixed, t_viz, rpm_viz).cpu().numpy().reshape(nx_viz, ny_viz)
        T_plot = T_room + DeltaT_r * theta
    
    ax.clear()
    cf = ax.contourf(Xv * L, Yv * L, T_plot, levels=levels, cmap="inferno")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    # Add boundaries
    ax.plot(shoulder_right_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    ax.plot(shoulder_left_x, shoulder_y, color='white', linestyle='--', linewidth=2)
    rect = patches.Rectangle((pin_left, pin_bottom), pin_width, Hp_raw, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    return cf,

ani = FuncAnimation(fig, update, frames=len(t_frames), interval=100)
gif_path = results_dir / "fixed_z_animation.gif"
ani.save(gif_path, writer=PillowWriter(fps=10))
plt.show()
net.train()