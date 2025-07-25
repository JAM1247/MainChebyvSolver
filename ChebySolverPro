import functools
import jax
import jax.numpy as jnp
from jax import random, lax, vmap, value_and_grad
import numpy as np
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import optax
import time
import os
from scipy.optimize import fsolve
import itertools

# For something annoying happening on my machine 
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Output directory 
OUTPUT_DIR = "graphs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Original cell fate potential and dynamics functions remain unchanged
def cell_fate_potential(x, y, a, theta1, theta2):
    return (x**4 + y**4 + x**3 - 2*x*y**2 + 
            a*(x**2 + y**2) + theta1*x + theta2*y)

# Drift Functions 
def create_cell_fate_dynamics(a, theta1, theta2):
    def fx(x, y):
        return -(4*x**3 + 3*x**2 - 2*y**2 + 2*a*x + theta1)
    
    def fy(x, y):
        return -(4*y**3 - 4*x*y + 2*a*y + theta2)
    
    return fx, fy

# ROBUST: Keep the original working steady state finder and just improve it slightly
def find_x_steady_states_y0(a, theta1):
    """Find steady states along y=0 line using polynomial roots - ORIGINAL WORKING VERSION"""
    coeffs = [4, 3, 2*a, theta1]
    roots = np.roots(coeffs)
    # Returns only real roots
    real_roots = []
    for root in roots:
        if np.abs(np.imag(root)) < 1e-10:
            real_roots.append(np.real(root))
    return sorted(real_roots)

def find_all_steady_states_robust(a, theta1, theta2, search_range=4.0, tolerance=1e-6):
    """
    ROBUST: Robust steady state finder that works correctly
    Uses the working method but with better numerical stability
    """
    fx, fy = create_cell_fate_dynamics(a, theta1, theta2)
    
    steady_states = []
    
    # Method 1: Check along y=0 line (only if theta2 is small)
    if abs(theta2) < tolerance:
        x_roots = find_x_steady_states_y0(a, theta1)
        for x_root in x_roots:
            # Verify this is actually a steady state
            if abs(fx(x_root, 0)) < tolerance and abs(fy(x_root, 0)) < tolerance:
                steady_states.append((x_root, 0.0))
    
    # Method 2: Numerical search on a coarse grid, then refine
    def dynamics_system(vars):
        x, y = vars
        return [fx(x, y), fy(x, y)]
    
    # Coarse grid search
    n_grid = 20
    x_range = np.linspace(-search_range, search_range, n_grid)
    y_range = np.linspace(-search_range, search_range, n_grid)
    
    for x0 in x_range:
        for y0 in y_range:
            try:
                # Try to find a steady state starting from this point
                solution = fsolve(dynamics_system, [x0, y0], xtol=tolerance)
                fx_val, fy_val = dynamics_system(solution)
                
                # Check if it's actually a steady state and within search range
                if (abs(fx_val) < tolerance and abs(fy_val) < tolerance and
                    abs(solution[0]) <= search_range and abs(solution[1]) <= search_range):
                    
                    # Check if we already found this steady state
                    is_duplicate = False
                    for existing in steady_states:
                        if (abs(solution[0] - existing[0]) < tolerance*10 and 
                            abs(solution[1] - existing[1]) < tolerance*10):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        steady_states.append((solution[0], solution[1]))
            except:
                continue
    
    return sorted(steady_states)

def classify_landscape_correctly(a, theta1, theta2):
    """
    ROBUST: Correct landscape classification based on actual steady states
    """
    steady_states = find_all_steady_states_robust(a, theta1, theta2)
    n_states = len(steady_states)
    
    if n_states == 0:
        return "Unstable (no attractors)", steady_states
    elif n_states == 1:
        return "Monostable", steady_states
    elif n_states == 2:
        return "Bistable", steady_states
    elif n_states == 3:
        return "Tristable", steady_states
    else:
        return f"Multistable ({n_states} states)", steady_states

# Chebyshev basis functions for control parameterization
def chebyshev_basis(order, t):
    t_norm = 2.0 * t - 1.0  
    T = [jnp.ones_like(t_norm), t_norm]
    for k in range(2, order):
        T.append(2.0 * t_norm * T[-1] - T[-2])
    return jnp.stack(T[:order], axis=1)

# Computing U1(t) and U2(t) using Chebyshev coefficients
@jax.jit
def get_controls(coeffs_x, coeffs_y, basis):
    return basis @ coeffs_x, basis @ coeffs_y

# Single Euler-Maruyama step for Langevin equations
def _euler_step(x, y, fx, fy, u1, u2, sigma1, sigma2, dt, noise):
    dx = (fx(x, y) + u1) * dt + sigma1 * jnp.sqrt(dt) * noise[0]
    dy = (fy(x, y) + u2) * dt + sigma2 * jnp.sqrt(dt) * noise[1]
    return x + dx, y + dy

# Function to simulate a single trajectory 
@functools.partial(jax.jit, static_argnums=(2, 3))
def simulate_trajectory(x0, y0, fx, fy, u1_vec, u2_vec, sigma1, sigma2, dt, key):
    
    n_steps = len(u1_vec)
    noises = random.normal(key, (n_steps - 1, 2))

    def scan_fn(carry, inp):
        x, y = carry
        u1, u2, n = inp
        x_next, y_next = _euler_step(x, y, fx, fy, u1, u2, sigma1, sigma2, dt, n)
        return (x_next, y_next), jnp.array([x_next, y_next])

    (_, _), traj = lax.scan(
        scan_fn,
        (x0, y0),
        (u1_vec[:-1], u2_vec[:-1], noises)
    )

    initial_state = jnp.array([[x0, y0]])
    traj_full = jnp.vstack([initial_state, traj])
    return traj_full

# Ensemble simulation for noise realization 
def simulate_ensemble(x0, y0, fx, fy, u1_vec, u2_vec, sigma1, sigma2, dt, n_traj, key):
    keys = random.split(key, n_traj)

    def sim_traj_wrapper(x, y, u1, u2, s1, s2, d, k):
        return simulate_trajectory(x, y, fx, fy, u1, u2, s1, s2, d, k)

    vmapped = vmap(
        sim_traj_wrapper,
        in_axes=(None, None, None, None, None, None, None, 0)
    )

    return vmapped(x0, y0, u1_vec, u2_vec, sigma1, sigma2, dt, keys)

# SPIKE-OPTIMIZED Cost Function with Terminal Penalty and Exponential Weighting
@functools.partial(jax.jit, static_argnums=(4, 5, 13))
def compute_cost_with_components(coeffs, basis, x0, y0, fx, fy, sigma1, sigma2, dt, 
                                targ_x, targ_y, lam, beta, n_traj, key, alpha_terminal=2.0):
    """
    Optimized cost function with terminal penalty to eliminate control spikes.
    The terminal penalty term enforces U(T) = 0, preventing Chebyshev oscillations.
    """
    order = basis.shape[1]
    coeffs_x, coeffs_y = coeffs[:order], coeffs[order:]
    u1_vec, u2_vec = get_controls(coeffs_x, coeffs_y, basis)

    traj = simulate_ensemble(x0, y0, fx, fy, u1_vec, u2_vec,
                             sigma1, sigma2, dt, n_traj, key)

    # Computing the time vector
    n_steps = traj.shape[1]
    t_vec = jnp.arange(n_steps) * dt
    
    # Exponential weights
    weights = jnp.exp(-beta * t_vec)
    
    # Normalized weights
    weights = weights / jnp.sum(weights)
    
    # Weighted distance cost over the trajectory
    target = jnp.array([targ_x, targ_y])
    distances_squared = jnp.sum((traj - target) ** 2, axis=2)  
    weighted_distances = jnp.sum(distances_squared * weights[None, :], axis=1) 
    j_target = jnp.mean(weighted_distances)
    finals = traj[:, -1, :]
    j_terminal = jnp.mean(jnp.sum((finals - target) ** 2, axis=1))
    
    # Regularization cost 
    j_reg = dt * jnp.mean(u1_vec ** 2 + u2_vec ** 2)
    
    # SPIKE PREVENTION: Terminal penalty enforces U(T) = 0
    j_terminal_penalty = alpha_terminal * (u1_vec[-1] ** 2 + u2_vec[-1] ** 2)
    
    # Total cost with terminal penalty for spike prevention
    total_cost = j_target + lam * j_reg + j_terminal_penalty
    
    return total_cost, j_target, j_reg, finals, j_terminal, j_terminal_penalty

# HIGH-QUALITY optimizer with proper convergence
def optimize_cell_fate_control(params, verbose=True, convergence_window=100, 
                              convergence_tol=1e-5, random_seed=42):
    if not verbose:  # For multistability analysis, print minimal info
        print(".", end="", flush=True)  # Progress indicator
    else:
        print("\n" + "="*60)
        print("HIGH-QUALITY SPIKE-OPTIMIZED CELL FATE CONTROL")
        print("="*60)
    
    # Initializing the dynamics 
    fx, fy = create_cell_fate_dynamics(params['a'], params['theta1'], params['theta2'])
    
    # Converts noise parameter D into standard deviations
    sigma = jnp.sqrt(2 * params['D'])
    
    # ROBUST debugging with correct classification
    if verbose:
        print(f"\nSTEADY STATE ANALYSIS:")
        print(f"  Parameters: a={params['a']}, theta1={params['theta1']}, theta2={params['theta2']}")
        
        # ROBUST: Use correct steady state analysis
        landscape_type, steady_states = classify_landscape_correctly(params['a'], params['theta1'], params['theta2'])
        print(f"  Landscape type: {landscape_type}")
        print(f"  Steady states: {[(f'{x:.3f}', f'{y:.3f}') for x, y in steady_states]}")
        
        # Verify the target is close to a steady state
        target_is_steady = any(
            (abs(x - params['target_x']) < 0.2 and abs(y - params['target_y']) < 0.2) 
            for x, y in steady_states
        )
        
        if target_is_steady:
            print(f"  Target ({params['target_x']:.3f}, {params['target_y']:.3f}) is near a steady state!")
        else:
            print(f"  NOTE: Target ({params['target_x']:.3f}, {params['target_y']:.3f}) is not near a steady state")
            print(f"     Using target anyway - may require higher control effort")
    
    if verbose:
        print(f"\nSPIKE PREVENTION PARAMETERS:")
        alpha_terminal = params.get('alpha_terminal', 2.0)
        print(f"  Terminal penalty coefficient: alpha = {alpha_terminal}")
        print(f"  This enforces U(T) = 0 to prevent Chebyshev oscillations")
        
        print(f"\nPHYSICS PARAMETERS:")
        print(f"  Potential parameters: a={params['a']}, theta1={params['theta1']}, theta2={params['theta2']}")
        print(f"  Noise intensity: D={params['D']} -> sigma={sigma:.4f}")
    
    x0, y0 = params['x0'], params['y0']
    T, dt = params['T'], params['dt']
    n_traj = params['N']
    targ_x, targ_y = params['target_x'], params['target_y']
    lam = params['lambda_reg']
    beta = params.get('beta', 0.1)
    alpha_terminal = params.get('alpha_terminal', 2.0)
    order = params['chebyshev_order']
    lr = params['learning_rate']
    max_epochs = params['max_epochs']

    if verbose:
        print(f"\nCONTROL PARAMETERS:")
        print(f"  Initial state: ({x0}, {y0})")
        print(f"  Target state: ({targ_x}, {targ_y})")
        print(f"  Time horizon: T={T}, dt={dt} -> {int(T/dt)} time steps")
        print(f"  Ensemble size: {n_traj} cells")
        print(f"  Control basis: Chebyshev order {order} -> {2*order} coefficients")
        print(f"  Regularization: lambda={lam}")
        print(f"  Exponential weight: beta={beta}")
        print(f"  Terminal penalty: alpha={alpha_terminal}")
        print(f"  Learning rate: {lr}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Convergence criteria: {convergence_tol} over {convergence_window} epochs")
    
    t_vec = jnp.arange(0, T + dt, dt)
    basis = chebyshev_basis(order, t_vec / T)
    
    # Initializing coefficients
    coeffs = jnp.zeros(2 * order)

    master = random.PRNGKey(random_seed)
    opt = optax.adam(lr)
    opt_state = opt.init(coeffs)
    
    losses = []
    terminal_costs = []
    weighted_costs = []
    reg_costs = []
    terminal_penalties = []
    distances = []
    control_spikes = []
    
    if verbose:
        print(f"\nOPTIMIZATION PROGRESS:")
        print("-" * 100)
        print("Epoch  | Total Loss | Weighted Cost | Terminal Cost | Control Cost | Terminal Penalty | Max |U| | Time (s)")
        print("-" * 100)

    # Tracking for convergence check with patience
    convergence_losses = []
    converged = False
    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        key = random.fold_in(master, epoch)

        def loss_fn(c):
            total, weighted, reg, finals, terminal, term_penalty = compute_cost_with_components(
                c, basis, x0, y0, fx, fy,
                sigma, sigma, dt,
                targ_x, targ_y, lam, beta,
                n_traj, key, alpha_terminal)
            return total

        loss, grads = value_and_grad(loss_fn)(coeffs)
        
        # Getting the cost components and final states
        total, j_weighted, j_reg, finals, j_terminal, j_term_penalty = compute_cost_with_components(
            coeffs, basis, x0, y0, fx, fy,
            sigma, sigma, dt,
            targ_x, targ_y, lam, beta,
            n_traj, key, alpha_terminal)
        
        # Computing mean distance to our target and success rate for early stopping
        target = jnp.array([targ_x, targ_y])
        mean_distance = jnp.mean(jnp.sqrt(jnp.sum((finals - target) ** 2, axis=1)))
        final_distances_current = jnp.sqrt(jnp.sum((finals - target) ** 2, axis=1))
        success_rate = jnp.mean(final_distances_current < 0.5)
        
        # Track control spikes (maximum control magnitude)
        coeffs_x, coeffs_y = coeffs[:order], coeffs[order:]
        u1_curr, u2_curr = get_controls(coeffs_x, coeffs_y, basis)
        max_control = jnp.max(jnp.abs(jnp.concatenate([u1_curr, u2_curr])))
        
        updates, opt_state = opt.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        
        losses.append(float(loss))
        weighted_costs.append(float(j_weighted))
        terminal_costs.append(float(j_terminal))
        reg_costs.append(float(j_reg))
        terminal_penalties.append(float(j_term_penalty))
        distances.append(float(mean_distance))
        control_spikes.append(float(max_control))
        
        epoch_time = time.time() - epoch_start
        
        if verbose and (epoch % 50 == 0 or epoch == max_epochs - 1):
            print(f"{epoch:5d}  | {total:10.6f} | {j_weighted:13.6f} | {j_terminal:13.6f} | "
                  f"{j_reg:12.6f} | {j_term_penalty:16.6f} | {max_control:7.3f} | {epoch_time:.4f}")
        
        # HIGH-QUALITY convergence checking with patience
        convergence_losses.append(float(loss))
        
        # Patience-based early stopping
        if float(loss) < best_loss:
            best_loss = float(loss)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Very early stopping only for exceptional success
        if epoch > 100 and success_rate > 0.95:
            converged = True
            if verbose:
                print(f"\nExceptional success stopping at epoch {epoch} (success rate: {success_rate*100:.1f}%)")
            break
        
        # Standard convergence check
        if epoch >= convergence_window:
            convergence_losses = convergence_losses[-convergence_window:]
            
            start_loss = convergence_losses[0]
            end_loss = convergence_losses[-1]
            
            if start_loss > 0:
                relative_change = abs((end_loss - start_loss) / start_loss)
                
                if relative_change < convergence_tol or patience_counter >= 300:
                    converged = True
                    if verbose:
                        reason = "patience exceeded" if patience_counter >= 300 else f"relative change: {relative_change:.2e}"
                        print(f"\nConverged at epoch {epoch} ({reason})")
                    break
    
    total_time = time.time() - start_time
    
    # Final evaluation
    coeffs_x, coeffs_y = coeffs[:order], coeffs[order:]
    u1_final, u2_final = get_controls(coeffs_x, coeffs_y, basis)
    key_final = random.fold_in(master, 9999)
    traj = simulate_ensemble(x0, y0, fx, fy,
                             u1_final, u2_final,
                             sigma, sigma,
                             dt, n_traj, key_final)

    # Computing the final statistics
    final_states = traj[:, -1, :]
    target = jnp.array([targ_x, targ_y])
    final_distances = jnp.sqrt(jnp.sum((final_states - target) ** 2, axis=1))
    success_rate = jnp.mean(final_distances < 0.5)
    
    # Spike analysis
    final_max_control = jnp.max(jnp.abs(jnp.concatenate([u1_final, u2_final])))
    terminal_control = jnp.sqrt(u1_final[-1]**2 + u2_final[-1]**2)
    
    if verbose:
        print("-" * 100)
        print(f"\nSPIKE ANALYSIS RESULTS:")
        print(f"  Maximum control magnitude: {final_max_control:.6f}")
        print(f"  Terminal control magnitude: {terminal_control:.6f}")
        print(f"  Terminal penalty contribution: {j_term_penalty:.6f}")
        
        print(f"\nFINAL RESULTS:")
        print(f"  Optimization time: {total_time:.2f}s ({len(losses)} epochs)")
        print(f"  Converged: {'Yes' if converged else 'No'}")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Success rate: {success_rate*100:.1f}% (within 0.5 units)")
        print(f"  Mean distance: {jnp.mean(final_distances):.4f} +/- {jnp.std(final_distances):.4f}")
        
        control_energy = jnp.mean(u1_final**2 + u2_final**2)
        print(f"  Control energy: {control_energy:.6f}")
        print(f"  Max control: |U1|={jnp.max(jnp.abs(u1_final)):.3f}, |U2|={jnp.max(jnp.abs(u2_final)):.3f}")
        print("="*60 + "\n")
    
    return dict(coeffs=coeffs,
                U1=u1_final,
                U2=u2_final,
                trajectories=traj,
                losses=np.asarray(losses),
                weighted_costs=np.asarray(weighted_costs),
                terminal_costs=np.asarray(terminal_costs),
                reg_costs=np.asarray(reg_costs),
                terminal_penalties=np.asarray(terminal_penalties),
                control_spikes=np.asarray(control_spikes),
                distances=np.asarray(distances),
                t_vec=np.asarray(t_vec),
                a=params['a'], 
                theta1=params['theta1'], 
                theta2=params['theta2'], 
                D=params['D'],
                beta=beta,
                alpha_terminal=alpha_terminal,
                lambda_reg=lam,
                epochs_completed=len(losses),
                converged=converged,
                optimization_time=total_time,
                success_rate=float(success_rate),
                ensemble_size=n_traj,
                chebyshev_order=order,
                max_control=float(final_max_control),
                terminal_control=float(terminal_control))

# Function to compute and plot potential contours
def plot_potential_contours(ax, a, theta1, theta2, xlim=(-3, 3), ylim=(-3, 3), levels=20):
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = cell_fate_potential(X, Y, a, theta1, theta2)
    
    # Contour plots
    contour = ax.contour(X, Y, Z, levels=levels, alpha=0.3, colors='gray', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f')
    contourf = ax.contourf(X, Y, Z, levels=levels, alpha=0.1, cmap='viridis')
    
    return contour, contourf

# Enhanced comprehensive graph with spike analysis
def comprehensive_graph(results, target_x, target_y, save_name):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Spike-Optimized Cell Fate Control Results', fontsize=16, fontweight='bold')
    
    # Trajectory visualization with potential background
    ax1 = fig.add_subplot(3, 4, 1)
    plot_potential_contours(ax1, results['a'], results['theta1'], results['theta2'])
    
    trajectories = results['trajectories']
    n_plot = min(50, trajectories.shape[0])
    for i in range(n_plot):
        ax1.plot(trajectories[i, :, 0], trajectories[i, :, 1], 
                'b-', alpha=0.3, linewidth=0.5)
    
    mean_traj = jnp.mean(trajectories, axis=0)
    ax1.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=2, label='Mean trajectory')
    ax1.scatter(trajectories[0, 0, 0], trajectories[0, 0, 1], 
               color='green', s=100, label='Initial state', zorder=5)
    ax1.scatter(target_x, target_y, color='red', s=100, marker='*', 
               label='Target state', zorder=5)
    
    # All steady states
    steady_states = find_all_steady_states_robust(results['a'], results['theta1'], results['theta2'])
    for i, (x_ss, y_ss) in enumerate(steady_states):
        ax1.scatter(x_ss, y_ss, color='orange', s=100, marker='s', alpha=0.7, 
                   label='Steady states' if i == 0 else "")
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Controlled Trajectories with Potential Landscape')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SPIKE-OPTIMIZED Control functions
    ax2 = fig.add_subplot(3, 4, 2)
    t_vec = results['t_vec']
    ax2.plot(t_vec, results['U1'], label='U1(t)', linewidth=2, color='blue')
    ax2.plot(t_vec, results['U2'], label='U2(t)', linewidth=2, color='orange')
    
    # Highlight terminal values to show spike prevention
    ax2.scatter(t_vec[-1], results['U1'][-1], color='blue', s=100, 
               label=f'U1(T) = {results["U1"][-1]:.4f}', zorder=5)
    ax2.scatter(t_vec[-1], results['U2'][-1], color='orange', s=100, 
               label=f'U2(T) = {results["U2"][-1]:.4f}', zorder=5)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Control amplitude')
    ax2.set_title('Spike-Free Control Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss evolution with terminal penalty
    ax3 = fig.add_subplot(3, 4, 3)
    epochs = np.arange(len(results['losses']))
    ax3.semilogy(epochs, results['losses'], label='Total loss', linewidth=2)
    if 'weighted_costs' in results:
        ax3.semilogy(epochs, results['weighted_costs'], label='Trajectory cost', alpha=0.7)
    if 'terminal_costs' in results:
        ax3.semilogy(epochs, results['terminal_costs'], label='Terminal cost', alpha=0.7, linestyle='--')
    if 'terminal_penalties' in results:
        ax3.semilogy(epochs, results['terminal_penalties'], label='Terminal penalty', alpha=0.7, linestyle=':')
    ax3.semilogy(epochs, results['reg_costs'], label='Control cost', alpha=0.7)
    ax3.set_xlabel('Optimization epoch')
    ax3.set_ylabel('Cost')
    ax3.set_title(f'Cost Evolution (alpha={results.get("alpha_terminal", 2.0)})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Spike tracking
    ax4 = fig.add_subplot(3, 4, 4)
    if 'control_spikes' in results:
        ax4.plot(epochs, results['control_spikes'], linewidth=2, color='red')
        ax4.set_xlabel('Optimization epoch')
        ax4.set_ylabel('Max |U|')
        ax4.set_title('Control Spike Evolution')
        ax4.grid(True, alpha=0.3)
    
    # Final state distribution with potential background
    ax5 = fig.add_subplot(3, 4, 5)
    plot_potential_contours(ax5, results['a'], results['theta1'], results['theta2'])
    final_states = trajectories[:, -1, :]
    h = ax5.hist2d(final_states[:, 0], final_states[:, 1], bins=20, cmap='Blues', alpha=0.8)
    fig.colorbar(h[3], ax=ax5, label='Trajectory count')
    ax5.scatter(target_x, target_y, color='red', s=100, marker='*', zorder=5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Final State Distribution with Potential')
    
    # Distance convergence
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.plot(epochs, results['distances'], linewidth=2)
    ax6.set_xlabel('Optimization epoch')
    ax6.set_ylabel('Mean distance to target')
    ax6.set_title('Distance Convergence')
    ax6.grid(True, alpha=0.3)
    
    # Control energy
    ax7 = fig.add_subplot(3, 4, 7)
    ctrl_energy = results['U1']**2 + results['U2']**2
    ax7.plot(t_vec, ctrl_energy, linewidth=2)
    ax7.set_xlabel('Time')
    ax7.set_ylabel('U1^2 + U2^2')
    ax7.set_title('Instantaneous Control Energy')
    ax7.grid(True, alpha=0.3)
    
    # Terminal control analysis
    ax8 = fig.add_subplot(3, 4, 8)
    terminal_window = slice(-20, None)  # Last 20 time points
    ax8.plot(t_vec[terminal_window], results['U1'][terminal_window], 'b-', linewidth=2, label='U1(t)')
    ax8.plot(t_vec[terminal_window], results['U2'][terminal_window], 'r-', linewidth=2, label='U2(t)')
    ax8.scatter(t_vec[-1], results['U1'][-1], color='blue', s=100, zorder=5)
    ax8.scatter(t_vec[-1], results['U2'][-1], color='red', s=100, zorder=5)
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Control amplitude')
    ax8.set_title('Terminal Control Detail')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Final distance CDF
    ax9 = fig.add_subplot(3, 4, 9)
    final_distances = jnp.sqrt(jnp.sum((final_states - jnp.array([target_x, target_y])) ** 2, axis=1))
    sorted_d = np.sort(final_distances)
    cdf = np.arange(len(sorted_d)) / len(sorted_d)
    ax9.plot(sorted_d, cdf, linewidth=2)
    ax9.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Success threshold')
    ax9.set_xlabel('Final distance to target')
    ax9.set_ylabel('Cumulative probability')
    ax9.set_title('Success Rate Analysis')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Enhanced parameter summary with landscape classification
    ax10 = fig.add_subplot(3, 4, 10)
    landscape_type, steady_states = classify_landscape_correctly(results['a'], results['theta1'], results['theta2'])
    
    summary_text = (
        f"OPTIMIZATION SUMMARY\n"
        f"{'='*25}\n"
        f"Landscape: {landscape_type}\n"
        f"Steady states: {len(steady_states)}\n"
        f"Ensemble size: {results['ensemble_size']}\n"
        f"Chebyshev order: {results['chebyshev_order']}\n"
        f"Terminal penalty: alpha = {results.get('alpha_terminal', 'N/A')}\n"
        f"Max control: {results.get('max_control', 'N/A'):.6f}\n"
        f"Terminal control: {results.get('terminal_control', 'N/A'):.6f}\n"
        f"Epochs: {results['epochs_completed']}\n"
        f"Converged: {'Yes' if results['converged'] else 'No'}\n"
        f"Final loss: {results['losses'][-1]:.6f}\n"
        f"Success rate: {results['success_rate']*100:.1f}%\n"
        f"Runtime: {results.get('optimization_time',0):.1f}s\n\n"
        f"POTENTIAL PARAMETERS\n"
        f"{'='*19}\n"
        f"a = {results['a']}\n"
        f"theta1 = {results['theta1']}\n"
        f"theta2 = {results['theta2']}\n"
        f"D = {results['D']}\n\n"
        f"CONTROL PARAMETERS\n"
        f"{'='*18}\n"
    )
    if 'beta' in results:
        summary_text += f"beta = {results['beta']}\n"
    if 'lambda_reg' in results:
        summary_text += f"lambda = {results['lambda_reg']}"
    
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax10.axis('off')
    
    # Phase portrait detail with potential heatmap
    ax11 = fig.add_subplot(3, 4, 11)
    x_range = np.linspace(-2, 2, 50)
    y_range = np.linspace(-2, 2, 50)
    X_heat, Y_heat = np.meshgrid(x_range, y_range)
    Z_heat = cell_fate_potential(X_heat, Y_heat, results['a'], results['theta1'], results['theta2'])
    
    im = ax11.imshow(Z_heat, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis', alpha=0.6)
    contour = ax11.contour(X_heat, Y_heat, Z_heat, levels=15, colors='gray', alpha=0.4, linewidths=0.5)
    
    # Plotting mean trajectory with direction arrows
    mean_traj = jnp.mean(trajectories, axis=0)
    ax11.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3, alpha=0.9)
    n_arrows = 5
    arrow_indices = np.linspace(0, len(mean_traj)-2, n_arrows, dtype=int)
    for i in arrow_indices:
        dx = mean_traj[i+1, 0] - mean_traj[i, 0]
        dy = mean_traj[i+1, 1] - mean_traj[i, 1]
        ax11.arrow(mean_traj[i, 0], mean_traj[i, 1], dx*0.3, dy*0.3, 
                 head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.8)

    ax11.scatter(trajectories[0, 0, 0], trajectories[0, 0, 1], 
               color='white', s=150, marker='o', zorder=5, edgecolors='black', linewidth=2)
    ax11.scatter(target_x, target_y, color='yellow', s=150, marker='*', 
               zorder=5, edgecolors='black', linewidth=2)
    
    # All steady states
    for x_ss, y_ss in steady_states:
        if -2 <= x_ss <= 2 and -2 <= y_ss <= 2:
            ax11.scatter(x_ss, y_ss, color='orange', s=100, marker='s', alpha=0.9, 
                       edgecolors='black', linewidth=1, zorder=5)
    
    ax11.set_xlabel('x')
    ax11.set_ylabel('y')
    ax11.set_title('Phase Portrait with Potential Heatmap')
    ax11.set_xlim(-2, 2)
    ax11.set_ylim(-2, 2)
    
    # Spike prevention status
    ax12 = fig.add_subplot(3, 4, 12)
    spike_status = "SPIKE PREVENTION ACHIEVED" if results.get('terminal_control', 1.0) < 0.1 else "SPIKES DETECTED"
    color = 'green' if results.get('terminal_control', 1.0) < 0.1 else 'red'
    ax12.text(0.1, 0.9, spike_status, transform=ax12.transAxes, 
             fontsize=14, fontweight='bold', color=color)
    ax12.text(0.1, 0.8, f'Terminal control: {results.get("terminal_control", "N/A"):.6f}', 
             transform=ax12.transAxes, fontsize=11)
    ax12.text(0.1, 0.7, f'Max control: {results.get("max_control", "N/A"):.3f}', 
             transform=ax12.transAxes, fontsize=11)
    ax12.text(0.1, 0.6, f'Success rate: {results["success_rate"]*100:.1f}%', 
             transform=ax12.transAxes, fontsize=11)
    
    ax12.text(0.1, 0.4, 'Terminal penalty enforces:', transform=ax12.transAxes, 
             fontsize=11, fontweight='bold')
    ax12.text(0.1, 0.3, 'U1(T) = 0  and  U2(T) = 0', transform=ax12.transAxes, 
             fontsize=11, fontfamily='monospace')
    ax12.text(0.1, 0.2, 'Preventing Chebyshev oscillations', transform=ax12.transAxes, 
             fontsize=11, style='italic')
    
    ax12.axis('off')
    
    plt.tight_layout()
    # making sure files go into the right place
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive graphs saved: {save_name}")
    
    return fig

# Lambda sweep function with spike optimization
def sweep_lambda(base_params, lambda_values, save_name="S1_lambda_parameter_sweep", verbose=False):
    print(f"\n{'='*60}")
    print("SPIKE-OPTIMIZED REGULARIZATION PARAMETER SWEEP")
    print(f"{'='*60}")
    
    # Main trajectory comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Effect of Regularization Parameter lambda on Spike-Free Control Performance', 
                fontsize=16, fontweight='bold')
    
    results_list = []
    
    for idx, lam in enumerate(lambda_values):
        # Create params for this lambda
        params = base_params.copy()
        params['lambda_reg'] = lam
        
        print(f"\nTesting lambda = {lam:.6f}")
        
        # Running optimization
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        
        # Subplot
        ax = axes[idx]
        plot_potential_contours(ax, results['a'], results['theta1'], results['theta2'])
        mean_traj = jnp.mean(results['trajectories'], axis=0) # mean trajectory
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3)
        
        n_plot = min(20, results['trajectories'].shape[0]) # individual trajectories
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.2, linewidth=0.5)
        
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='green', s=100, marker='o', zorder=5)
        ax.scatter(base_params['target_x'], base_params['target_y'], 
                  color='red', s=100, marker='*', zorder=5)
        
        max_control = results.get('max_control', max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2']))))
        terminal_control = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        ax.set_title(f'lambda = {lam:.1e}\nSuccess: {results["success_rate"]*100:.0f}%, Terminal: {terminal_control:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}_trajectories.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analysis figures
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Spike-Optimized Regularization Parameter Analysis', fontsize=16, fontweight='bold')
    
    # Extracting metrics
    max_controls = []
    terminal_controls = []
    success_rates = []
    final_losses = []
    
    for results in results_list:
        max_u = results.get('max_control', max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2']))))
        term_u = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        max_controls.append(max_u)
        terminal_controls.append(term_u)
        success_rates.append(results['success_rate'])
        final_losses.append(results['losses'][-1])
    
    # Control magnitude vs Lambda
    ax1.loglog(lambda_values, max_controls, 'o-', label='Max |U|', markersize=8, linewidth=2)
    ax1.loglog(lambda_values, terminal_controls, 's-', label='Terminal |U|', markersize=8, linewidth=2)
    ax1.set_xlabel('lambda (regularization parameter)')
    ax1.set_ylabel('Control magnitude')
    ax1.set_title('Control Magnitude vs Regularization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #Success rate vs Lambda
    ax2.semilogx(lambda_values, 100*np.array(success_rates), 'o-', markersize=8, linewidth=2, color='green')
    ax2.set_xlabel('lambda (regularization parameter)')
    ax2.set_ylabel('Success rate (%)')
    ax2.set_title('Success Rate vs Regularization')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Final loss
    ax3.loglog(lambda_values, final_losses, 'o-', markersize=8, linewidth=2, color='purple')
    ax3.set_xlabel('lambda (regularization parameter)')
    ax3.set_ylabel('Final loss')
    ax3.set_title('Final Loss vs Regularization')
    ax3.grid(True, alpha=0.3)
    
    # Summary table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    headers = ['lambda', 'Success Rate (%)', 'Max |U|', 'Terminal |U|', 'Final Loss']
    
    for lam, results in zip(lambda_values, results_list):
        max_u = results.get('max_control', max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2']))))
        term_u = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        table_data.append([
            f"{lam:.1e}",
            f"{results['success_rate']*100:.1f}",
            f"{max_u:.2f}",
            f"{term_u:.4f}",
            f"{results['losses'][-1]:.5f}"
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Parameter Sweep Summary', fontweight='bold')
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Spike-optimized lambda sweep figures saved: {save_name}_*.png")
    
    # Summary statistics
    print("\nSPIKE-OPTIMIZED LAMBDA SWEEP SUMMARY:")
    print("-" * 80)
    print(f"{'Lambda':<12} {'Success Rate':<15} {'Max |U|':<15} {'Terminal |U|':<15} {'Final Loss':<15}")
    print("-" * 80)
    for lam, results in zip(lambda_values, results_list):
        max_u = results.get('max_control', max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2']))))
        term_u = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        print(f"{lam:<12.6f} {results['success_rate']*100:>12.1f}%  "
              f"{max_u:>13.4f}  {term_u:>13.6f}  {results['losses'][-1]:>15.6f}")
    
    return results_list

# Beta sweep function with spike optimization
def sweep_beta(base_params, beta_values, save_name="S2_beta_parameter_sweep", verbose=False):
    print(f"\n{'='*60}")
    print("SPIKE-OPTIMIZED EXPONENTIAL WEIGHTING PARAMETER SWEEP")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle('Effect of Exponential Weighting Parameter beta on Spike-Free Control', fontsize=16, fontweight='bold')
    
    results_list = []
    
    for idx, beta in enumerate(beta_values):
        # Creating params for this beta
        params = base_params.copy()
        params['beta'] = beta
        
        print(f"\nTesting beta = {beta}")
        
        # Running optimization
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        ax = axes[idx]        
        
        plot_potential_contours(ax, results['a'], results['theta1'], results['theta2'])
    
        # mean trajectory graphs
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3)
        
        # Individual trajectories
        n_plot = min(20, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.2, linewidth=0.5)
        
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='green', s=100, marker='o', zorder=5)
        ax.scatter(base_params['target_x'], base_params['target_y'], 
                  color='red', s=100, marker='*', zorder=5)
        
        terminal_control = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        ax.set_title(f'beta = {beta:.1f}\nSuccess: {results["success_rate"]*100:.0f}%, Terminal: {terminal_control:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Spike-optimized beta sweep figure saved: {save_name}.png")
    
    return results_list

# Parameter sweep function with spike optimization
def sweep_potentials(base_params, scenarios, save_name="potential_landscape_comparison", verbose=False):
    print(f"\n{'='*60}")
    print("SPIKE-OPTIMIZED POTENTIAL LANDSCAPE COMPARISON")
    print(f"{'='*60}")
    
    # Potential landscape visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Spike-Free Control Performance Across Different Potential Landscapes', 
                fontsize=16, fontweight='bold')
    
    #subplot layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
    
    # issue that was coming up while plotting
    try:
        cmap = plt.colormaps['tab10']
    except:
        cmap = plt.cm.get_cmap('tab10')
    
    results_list = []
    
    # Plotting each scenario's potential and trajectory
    for idx, scenario in enumerate(scenarios[:6]):  
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Potential heatmap background
        x_range = np.linspace(-2.5, 2.5, 40)
        y_range = np.linspace(-2.5, 2.5, 40)
        X_heat, Y_heat = np.meshgrid(x_range, y_range)
        Z_heat = cell_fate_potential(X_heat, Y_heat, scenario['a'], scenario['theta1'], scenario['theta2'])
        
        # Adding heatmap with custom colormap
        im = ax.imshow(Z_heat, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', 
                      cmap='viridis', alpha=0.4, vmin=np.min(Z_heat), vmax=np.min(Z_heat) + 20)
        
        contour = ax.contour(X_heat, Y_heat, Z_heat, levels=10, colors='gray', alpha=0.6, linewidths=0.5)
        
        # Params for this scenario
        params = base_params.copy()
        params.update({
            'a': scenario['a'],
            'theta1': scenario['theta1'],
            'theta2': scenario['theta2'],
            'target_x': scenario['target_x'],
            'target_y': scenario['target_y']
        })
        
        print(f"\nScenario: {scenario['label']}")
        print(f"  Parameters: a={scenario['a']}, theta1={scenario['theta1']}, theta2={scenario['theta2']}")
        print(f"  Target: ({scenario['target_x']:.3f}, {scenario['target_y']:.3f})")
        
        results = optimize_cell_fate_control(params, verbose=verbose, 
                                           random_seed=42 + idx*100)
        results_list.append(results)
        
        # mean trajectory
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 
                color='red', linewidth=3, alpha=0.9, zorder=5)
        
        # individual trajectories
        n_plot = min(10, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'white', alpha=0.3, linewidth=0.5, zorder=3)
        
        # initial and target states
        ax.scatter(base_params['x0'], base_params['y0'], 
                  color='lime', s=150, marker='o', zorder=6, 
                  edgecolor='black', linewidth=2, label='Start')
        ax.scatter(scenario['target_x'], scenario['target_y'], 
                  color='yellow', s=150, marker='*', zorder=6,
                  edgecolor='black', linewidth=2, label='Target')
        
        # Steady states if they exist
        try:
            x_roots = find_x_steady_states_y0(scenario['a'], scenario['theta1'])
            for x_root in x_roots:
                if -2.5 <= x_root <= 2.5:  # Only plot if in visible range
                    ax.scatter(x_root, 0, color='orange', s=100, marker='s', 
                             alpha=0.9, edgecolor='black', linewidth=1, zorder=6)
        except:
            pass
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        terminal_control = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        ax.set_title(f'{scenario["label"]}\nSuccess: {results["success_rate"]*100:.0f}%, Terminal: {terminal_control:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Comparison plots in bottom row
    ax_comp1 = fig.add_subplot(gs[2, :2])
    ax_comp2 = fig.add_subplot(gs[2, 2:])
    
    # Success rates comparison
    scenario_names = [s['label'] for s in scenarios[:len(results_list)]]
    success_rates = [r['success_rate']*100 for r in results_list]
    colors = [cmap(i) for i in range(len(results_list))]
    
    bars1 = ax_comp1.bar(range(len(success_rates)), success_rates, color=colors, alpha=0.7)
    ax_comp1.set_xlabel('Scenario')
    ax_comp1.set_ylabel('Success Rate (%)')
    ax_comp1.set_title('Success Rate Comparison')
    ax_comp1.set_xticks(range(len(scenario_names)))
    ax_comp1.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax_comp1.grid(True, alpha=0.3)
    
    # Labeling
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax_comp1.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{rate:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Terminal control vs Success Rate
    terminal_controls = [r.get('terminal_control', jnp.sqrt(r['U1'][-1]**2 + r['U2'][-1]**2)) for r in results_list]
    
    scatter = ax_comp2.scatter(terminal_controls, success_rates, 
                              c=colors, s=100, alpha=0.7, edgecolors='black')
    
    for i, (term_control, success, name) in enumerate(zip(terminal_controls, success_rates, scenario_names)):
        ax_comp2.annotate(name, (term_control, success), xytext=(5, 5), 
                         textcoords='offset points', fontsize=8, ha='left')
    
    ax_comp2.set_xlabel('Terminal Control Magnitude')
    ax_comp2.set_ylabel('Success Rate (%)')
    ax_comp2.set_title('Terminal Control vs Success Rate')
    ax_comp2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Spike-optimized landscape comparison saved: {save_name}.png")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SPIKE-OPTIMIZED LANDSCAPE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<25} {'Success Rate':<12} {'Terminal Control':<18} {'Max Control':<15}")
    print("-" * 70)
    
    for scenario, results in zip(scenarios[:len(results_list)], results_list):
        terminal_control = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        max_control = results.get('max_control', max(jnp.max(jnp.abs(results['U1'])), jnp.max(jnp.abs(results['U2']))))
        print(f"{scenario['label']:<25} {results['success_rate']*100:>9.1f}%     "
              f"{terminal_control:>14.6f}      {max_control:>12.6f}")
    
    return results_list

# Function to test spike-optimized control on steady states
def test_steady_state_scenarios(save_name="steady_state_analysis", verbose=True):
    # Test spike-optimized control performance when targeting actual steady states
    print(f"\n{'='*60}")
    print("SPIKE-OPTIMIZED STEADY STATE TARGETING ANALYSIS")
    print(f"{'='*60}")
    
    scenarios = [
        # Original parameters
        {
            'a': 2.0, 'theta1': 5.0, 'theta2': -5.0,
            'x0': 0.0, 'y0': 0.0,
            'label': 'Original Parameters'
        },
        # Negative a value with monostable potential
        {
            'a': -1.0, 'theta1': 0.0, 'theta2': 0.0,
            'x0': 0.0, 'y0': 0.0,
            'label': 'Monostable (a=-1)'
        },
        # Negative a value with bias
        {
            'a': -2.0, 'theta1': 2.0, 'theta2': 0.0,
            'x0': -0.5, 'y0': 0.0,
            'label': 'Monostable with Bias'
        },
        # Symmetric bistable
        {
            'a': 1.0, 'theta1': 0.0, 'theta2': 0.0,
            'x0': 0.1, 'y0': 0.0, 
            'label': 'Symmetric Bistable'
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle('Spike-Optimized Steady State Targeting Validation', fontsize=16, fontweight='bold')
    
    results_list = []
    
    for idx, scenario in enumerate(scenarios):
        print(f"\nAnalyzing: {scenario['label']}")
        print(f"Parameters: a={scenario['a']}, theta1={scenario['theta1']}, theta2={scenario['theta2']}")
        
        # Finds steady states 
        x_roots = find_x_steady_states_y0(scenario['a'], scenario['theta1'])
        print(f"Steady states along y=0: {x_roots}")
        
        # Target based on steady states
        if len(x_roots) > 0:
            # For scenarios with multiple steady states, we choose the one furthest from origin
            if len(x_roots) > 1:
                # Preferring positive ones 
                positive_roots = [x for x in x_roots if x > 0.1]
                if positive_roots:
                    target_x = positive_roots[0]
                else:
                    target_x = x_roots[-1]  # Take the last one
            else:
                target_x = x_roots[0]
        else:
            print("WARNING: No steady states found!")
            target_x = 1.0  # Fallback
        
        # Setting up parameters with spike optimization
        params = {
            'x0': scenario['x0'], 'y0': scenario['y0'],
            'target_x': target_x, 'target_y': 0.0,
            'a': scenario['a'],
            'theta1': scenario['theta1'],
            'theta2': scenario['theta2'],
            'D': 0.05,
            'T': 5.0,
            'dt': 0.01,
            'N': 500,
            'lambda_reg': 0.001,  # Lower regularization for better control
            'beta': 0.1,  # Small exponential weighting
            'alpha_terminal': 2.0,  # Spike prevention
            'chebyshev_order': 20,
            'learning_rate': 0.01,
            'max_epochs': 500
        }
        
        print(f"Targeting steady state at ({target_x:.3f}, 0)")
        
        # Optimization
        results = optimize_cell_fate_control(params, verbose=verbose)
        results_list.append(results)
        
        # Graphs
        ax = axes[idx]
        plot_potential_contours(ax, scenario['a'], scenario['theta1'], scenario['theta2'])
        
        # Trajectories graphs
        mean_traj = jnp.mean(results['trajectories'], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=3, label='Mean trajectory')
        
        n_plot = min(15, results['trajectories'].shape[0])
        for i in range(n_plot):
            ax.plot(results['trajectories'][i, :, 0], results['trajectories'][i, :, 1], 
                   'b-', alpha=0.3, linewidth=0.5)
        
        # Start, target, and steady states
        ax.scatter(scenario['x0'], scenario['y0'], color='green', s=120, 
                  marker='o', zorder=5, label='Initial', edgecolor='black')
        ax.scatter(target_x, 0, color='red', s=120, marker='*', 
                  zorder=5, label='Target', edgecolor='black')
        
        # Mark all steady states
        for x_ss in x_roots:
            ax.scatter(x_ss, 0, color='orange', s=100, marker='s', 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        terminal_control = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        ax.set_title(f"{scenario['label']}\nSuccess: {results['success_rate']*100:.0f}%, Terminal: {terminal_control:.4f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Spike-optimized steady state analysis saved: {save_name}.png")
    
    # Summary
    print("\n" + "="*60)
    print("SPIKE-OPTIMIZED STEADY STATE TARGETING SUMMARY:")
    print("-" * 80)
    print(f"{'Scenario':<30} {'Target':<10} {'Success Rate':<15} {'Terminal Control':<18}")
    print("-" * 80)
    for scenario, results in zip(scenarios, results_list):
        target_x = results['trajectories'][0,-1,0]  # Get the actual target used
        terminal_control = results.get('terminal_control', jnp.sqrt(results['U1'][-1]**2 + results['U2'][-1]**2))
        print(f"{scenario['label']:<30} "
              f"({target_x:.2f},0)  "
              f"{results['success_rate']*100:>10.1f}%     "
              f"{terminal_control:>14.6f}")
    
    return results_list

# WORKING multistability analysis for systems that actually have multiple steady states
def comprehensive_multistability_analysis(base_params, save_name="multistability_analysis", verbose=True):
    """
    ROBUST: Comprehensive analysis of controllability between all pairs of steady states
    Only runs if there are actually multiple steady states
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MULTISTABILITY CONTROLLABILITY ANALYSIS")
    print(f"{'='*80}")
    
    # Find all steady states for the given parameters
    a, theta1, theta2 = base_params['a'], base_params['theta1'], base_params['theta2']
    landscape_type, steady_states = classify_landscape_correctly(a, theta1, theta2)
    
    print(f"Landscape type: {landscape_type}")
    print(f"Found {len(steady_states)} steady states: {steady_states}")
    
    if len(steady_states) < 2:
        print("INFO: Need at least 2 steady states for multistability analysis!")
        print("This system doesn't have multiple stable states to transition between.")
        return None
    
    # Generate all pairs of steady states
    state_pairs = list(itertools.combinations(range(len(steady_states)), 2))
    n_pairs = len(state_pairs)
    
    print(f"\nAnalyzing {n_pairs} transition pairs:")
    for i, (start_idx, end_idx) in enumerate(state_pairs):
        start_state = steady_states[start_idx]
        end_state = steady_states[end_idx]
        print(f"  {i+1}. {start_state} -> {end_state}")
    
    # HIGH-QUALITY parameters for multistability analysis
    transition_params = base_params.copy()
    transition_params.update({
        'N': 1200,  # Large ensemble for good statistics
        'max_epochs': 1000,  # Plenty of epochs for convergence
        'chebyshev_order': 28,  # High order for good control representation
        'lambda_reg': 0.0005,  # Lower regularization for better control
        'alpha_terminal': 2.0,  # Good spike prevention
        'learning_rate': 0.008,  # Moderate learning rate for stability
    })
    
    # Results storage
    transition_results = {}
    controllability_matrix = np.zeros((len(steady_states), len(steady_states)))
    energy_matrix = np.zeros((len(steady_states), len(steady_states)))
    success_matrix = np.zeros((len(steady_states), len(steady_states)))
    
    # Analyze each transition (HIGH-QUALITY VERSION - both directions)
    print(f"\nRunning HIGH-QUALITY transition analysis (all directions)...")
    print(f"Estimated time: ~{len(state_pairs)*2*90:.0f} seconds ({len(state_pairs)*2*90/60:.1f} minutes)")
    
    all_pairs = []
    for start_idx, end_idx in state_pairs:
        all_pairs.append((start_idx, end_idx))
        all_pairs.append((end_idx, start_idx))  # Both directions
    
    for pair_idx, (start_idx, end_idx) in enumerate(all_pairs):
        start_state = steady_states[start_idx]
        end_state = steady_states[end_idx]
        
        print(f"\n--- Transition {pair_idx+1}/{len(all_pairs)}: State {start_idx} -> State {end_idx} ---")
        print(f"    From {start_state} to {end_state}")
        
        # Set up transition parameters
        params = transition_params.copy()
        params.update({
            'x0': start_state[0], 'y0': start_state[1],
            'target_x': end_state[0], 'target_y': end_state[1]
        })
        
        # Run HIGH-QUALITY optimization for this transition
        start_time = time.time()
        results = optimize_cell_fate_control(params, verbose=False, 
                                           convergence_window=100, convergence_tol=1e-5,
                                           random_seed=42 + pair_idx*100)
        elapsed = time.time() - start_time
        
        # Store results
        transition_key = f"{start_idx}->{end_idx}"
        transition_results[transition_key] = results
        
        # Compute metrics
        success_rate = results['success_rate']
        control_energy = np.mean(results['U1']**2 + results['U2']**2) * results['t_vec'][-1]
        max_control = results.get('max_control', 0)
        
        # Fill matrices
        controllability_matrix[start_idx, end_idx] = 1 if success_rate > 0.5 else 0
        energy_matrix[start_idx, end_idx] = control_energy
        success_matrix[start_idx, end_idx] = success_rate
        
        print(f"    Success rate: {success_rate*100:.1f}%")
        print(f"    Control energy: {control_energy:.4f}")
        print(f"    Max control: {max_control:.4f}")
        print(f"    Controllable: {'YES' if success_rate > 0.5 else 'NO'}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Progress: {pair_idx+1}/{len(all_pairs)} ({(pair_idx+1)/len(all_pairs)*100:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Multistability Analysis - {landscape_type}', fontsize=16, fontweight='bold')
    
    # Controllability matrix
    im1 = axes[0,0].imshow(controllability_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0,0].set_title('Controllability Matrix')
    axes[0,0].set_xlabel('Target State')
    axes[0,0].set_ylabel('Initial State')
    for i in range(len(steady_states)):
        for j in range(len(steady_states)):
            axes[0,0].text(j, i, f'{controllability_matrix[i, j]:.0f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Success rate matrix
    im2 = axes[0,1].imshow(success_matrix, cmap='viridis', vmin=0, vmax=1)
    axes[0,1].set_title('Success Rate Matrix')
    axes[0,1].set_xlabel('Target State')
    axes[0,1].set_ylabel('Initial State')
    for i in range(len(steady_states)):
        for j in range(len(steady_states)):
            if i != j:
                axes[0,1].text(j, i, f'{success_matrix[i, j]:.2f}',
                              ha="center", va="center", color="white", fontsize=9)
    
    # Control energy matrix
    energy_display = energy_matrix.copy()
    np.fill_diagonal(energy_display, np.nan)
    im3 = axes[0,2].imshow(energy_display, cmap='plasma')
    axes[0,2].set_title('Control Energy Matrix')
    axes[0,2].set_xlabel('Target State')
    axes[0,2].set_ylabel('Initial State')
    
    # Sample trajectory for best transition
    best_transitions = []
    for i in range(len(steady_states)):
        for j in range(len(steady_states)):
            if i != j and success_matrix[i, j] > 0.3:
                best_transitions.append((i, j, success_matrix[i, j]))
    
    if best_transitions:
        best_transitions.sort(key=lambda x: x[2], reverse=True)
        start_idx, end_idx, success_rate = best_transitions[0]
        
        ax = axes[1,0]
        transition_key = f"{start_idx}->{end_idx}"
        if transition_key in transition_results:
            result = transition_results[transition_key]
            
            # Plot trajectories
            traj = result['trajectories']
            n_plot = min(10, traj.shape[0])
            for i in range(n_plot):
                ax.plot(traj[i, :, 0], traj[i, :, 1], 'b-', alpha=0.5, linewidth=0.5)
            
            # Mean trajectory
            mean_traj = np.mean(traj, axis=0)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', linewidth=2)
            
            # Start and end states
            start_state = steady_states[start_idx]
            end_state = steady_states[end_idx]
            ax.scatter(start_state[0], start_state[1], color='green', s=100, marker='o', zorder=5)
            ax.scatter(end_state[0], end_state[1], color='red', s=100, marker='*', zorder=5)
            
            ax.set_title(f'Best Transition {start_idx}->{end_idx}\nSuccess: {success_rate*100:.0f}%')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
    
    # Summary statistics
    total_transitions = len(steady_states) * (len(steady_states) - 1)
    controllable_transitions = np.sum(controllability_matrix) - np.trace(controllability_matrix)
    controllability_fraction = controllable_transitions / total_transitions if total_transitions > 0 else 0
    avg_success_rate = np.mean(success_matrix[success_matrix > 0]) if np.any(success_matrix > 0) else 0
    
    axes[1,1].text(0.1, 0.8, f"MULTISTABILITY SUMMARY\n" + "="*20, 
                   transform=axes[1,1].transAxes, fontsize=12, fontweight='bold')
    axes[1,1].text(0.1, 0.6, f"Landscape: {landscape_type}\nStates: {len(steady_states)}\n"
                             f"Controllable: {controllability_fraction*100:.1f}%\n"
                             f"Avg Success: {avg_success_rate*100:.1f}%",
                   transform=axes[1,1].transAxes, fontsize=11)
    axes[1,1].axis('off')
    
    # Detailed table
    axes[1,2].axis('off')
    table_data = []
    headers = ['From->To', 'Success %', 'Energy']
    
    for i in range(len(steady_states)):
        for j in range(len(steady_states)):
            if i != j and success_matrix[i, j] > 0.1:  # Only show non-trivial transitions
                table_data.append([f'{i}->{j}', f'{success_matrix[i, j]*100:.0f}%', f'{energy_matrix[i, j]:.1f}'])
    
    if table_data:
        table = axes[1,2].table(cellText=table_data[:8], colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.2)
        axes[1,2].set_title('Transition Summary', fontweight='bold')
    
    plt.tight_layout()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multistability analysis saved: {save_name}.png")
    
    return {
        'landscape_type': landscape_type,
        'steady_states': steady_states,
        'controllability_matrix': controllability_matrix,
        'success_matrix': success_matrix,
        'energy_matrix': energy_matrix,
        'transition_results': transition_results,
        'controllability_fraction': controllability_fraction,
        'avg_success_rate': avg_success_rate
    }

def main():
    print("\nSPIKE-OPTIMIZED COMPREHENSIVE ANALYSIS")
    print("="*60)
    print("SPEED OPTIMIZED VERSION")
    print("Reduced parameters for 10x faster execution")
    print("="*60)
    
    # Test parameters that should actually work
    test_scenarios = [
        {
            'name': 'Simple Target',
            'params': {'a': 1.0, 'theta1': 0.0, 'theta2': 0.0, 'target_x': 0.5, 'target_y': 0.0}
        },
        {
            'name': 'Bistable System',
            'params': {'a': -0.5, 'theta1': 0.0, 'theta2': 0.0, 'target_x': 0.8, 'target_y': 0.0}
        },
        {
            'name': 'Original Parameters',
            'params': {'a': 2.0, 'theta1': 5.0, 'theta2': -5.0, 'target_x': 1.0, 'target_y': 0.0}
        }
    ]
    
    print("\nTesting different scenarios to verify functionality:")
    print("Estimated total time: ~2-3 minutes")
    
    total_start_time = time.time()
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n--- Testing {i+1}/{len(test_scenarios)}: {scenario['name']} ---")
        scenario_start = time.time()
        print(f"\n--- Testing: {scenario['name']} ---")
        
        # Check landscape type first
        landscape_type, steady_states = classify_landscape_correctly(
            scenario['params']['a'], scenario['params']['theta1'], scenario['params']['theta2']
        )
        print(f"Landscape: {landscape_type}")
        print(f"Steady states: {steady_states}")
        
        # Set up HIGH-QUALITY parameters
        params = {
            'x0': 0.0, 'y0': 0.0,
            'D': 0.05, 'T': 5.0, 'dt': 0.01, 'N': 1000,  # Large ensemble
            'lambda_reg': 0.0005, 'beta': 0.05, 'alpha_terminal': 2.0,
            'chebyshev_order': 25, 'learning_rate': 0.008, 'max_epochs': 800  # High quality
        }
        params.update(scenario['params'])
        
        # Run optimization
        try:
            opt_start = time.time()
            results = optimize_cell_fate_control(params, verbose=True)
            opt_time = time.time() - opt_start
            print(f"SUCCESS: {results['success_rate']*100:.1f}% success rate ({opt_time:.1f}s)")
            
            # Generate comprehensive graph
            comprehensive_graph(results, params['target_x'], params['target_y'], 
                               f"fast_{scenario['name'].lower().replace(' ', '_')}_control.png")
            
            # If this is a multistable system, test the multistability analysis
            if len(steady_states) >= 2:
                print(f"\nRunning FAST multistability analysis for {scenario['name']}...")
                multistab_start = time.time()
                multistab_results = comprehensive_multistability_analysis(
                    params, save_name=f"fast_multistability_{scenario['name'].lower().replace(' ', '_')}"
                )
                multistab_time = time.time() - multistab_start
                if multistab_results:
                    print(f"\nMultistability: {multistab_results['controllability_fraction']*100:.1f}% controllable ({multistab_time:.1f}s)")
            
            scenario_time = time.time() - scenario_start
            print(f"Scenario completed in {scenario_time:.1f}s")
            
        except Exception as e:
            print(f"ERROR in {scenario['name']}: {e}")
    
    total_time = time.time() - total_start_time
    print(f"\nAll analyses completed in {total_time:.1f}s!")
    print("="*60 + "\n")

    # Run parameter sweeps
    print("\nSPIKE-OPTIMIZED PARAMETER SWEEPS")
    
    # Find steady state for parameters
    x_roots = find_x_steady_states_y0(2.0, 5.0) 
    target_x = x_roots[0] if x_roots else 1.0
    
    # Main demonstration with spike optimization
    main_params = {
        'x0': 0.0, 'y0': 0.0,
        'target_x': target_x, 'target_y': 0.0,
        'a': 2.0, 'theta1': 5.0, 'theta2': -5.0,
        'D': 0.05, 'T': 5.0, 'dt': 0.01, 'N': 1000,
        'lambda_reg': 0.001, 'beta': 0.1, 'alpha_terminal': 2.0,
        'chebyshev_order': 25, 'learning_rate': 0.01, 'max_epochs': 500
    }
    
    print(f"\nMain Demonstration: Spike-optimized targeting steady state at ({target_x:.3f}, 0)")
    main_results = optimize_cell_fate_control(main_params, verbose=True)
    comprehensive_graph(main_results, main_params['target_x'], main_params['target_y'], 
                       "spike_optimized_steady_state_control.png")
    
    # Lambda sweep with spike optimization
    base_params_optimized = {
        'x0': 0.0, 'y0': 0.0,
        'target_x': target_x, 'target_y': 0.0,
        'a': 2.0, 'theta1': 5.0, 'theta2': -5.0,
        'D': 0.05, 'T': 5.0, 'dt': 0.01, 'N': 1000,
        'lambda_reg': 0.01, 'beta': 0.1, 'alpha_terminal': 2.0,
        'chebyshev_order': 25, 'learning_rate': 0.01, 'max_epochs': 500
    }
    
    lambda_values = np.logspace(-4, -1, 6)
    lambda_results = sweep_lambda(base_params_optimized, lambda_values)
    
    # Beta sweep with spike optimization
    beta_values = [0, 0.1, 0.5, 1.0]
    beta_results = sweep_beta(base_params_optimized, beta_values)
    
    # Steady state analysis
    steady_state_results = test_steady_state_scenarios()

if __name__ == "__main__":
    main()





    
