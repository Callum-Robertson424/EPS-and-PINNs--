# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# tensorflow and numpy packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
# matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
# others for results
import time
import pandas as pd
import os

#-----------------------------------------------------------------------------#

        # Initializing the model #

#-----------------------------------------------------------------------------#

def energy_func(theta, omega):
    energy = 0.5 * (omega * omega) + 1 - np.cos(theta)
    return(energy)

#-----------------------------------------------------------------------------#

        # Plots #
                         
#-----------------------------------------------------------------------------#

def fade_plot():
    x = np.sin(theta_pred[:, 0])
    z = -np.cos(theta_pred[:, 0])
    
    xa = np.sin(theta)
    za = -np.cos(theta)
    
    num_positions = 6
    indices = np.linspace(0, 300, num_positions, dtype=int)

    colors = cm.Blues(np.linspace(0.1, 0.8, num_positions))
    colors2 = cm.Oranges(np.linspace(0.1, 0.8, num_positions)) 

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Trajectory Plots", fontsize=16, x=0.34)

    for i, idx in enumerate(indices):
        # Left plot (Predicted solution)
        axs[0].plot([0, x[idx]], [0, z[idx]], 'o-', color=colors[i], lw=2, alpha=0.8)
        axs[0].set_xlim(-1.5, 1.5)
        axs[0].set_ylim(-1.5, 1.5)
        axs[0].set_aspect('equal', 'box')
        axs[0].axhline(0, color='black', linewidth=0.5)
        axs[0].axvline(0, color='black', linewidth=0.5)
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Z')
        axs[0].set_title('Predicted Solution')

        # Right plot (Exact solution)
        axs[1].plot([0, xa[idx]], [0, za[idx]], 'o-', color=colors2[i], lw=2, alpha=0.8)
        axs[1].set_xlim(-1.5, 1.5)
        axs[1].set_ylim(-1.5, 1.5)
        axs[1].set_aspect('equal', 'box')
        axs[1].axhline(0, color='black', linewidth=0.5)
        axs[1].axvline(0, color='black', linewidth=0.5)
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')
        axs[1].set_title('Exact Solution')
        
    fig.add_artist(plt.Line2D([0.667, 0.667], [0, 1], color='black', lw=1, transform=fig.transFigure))
    
    axs[2].plot(theta_pred, omega_pred, lw=4, linestyle = "--", color='orange', label='Predicted phase')
    axs[2].plot(theta, omega, lw=1, color='blue', label="Exact Phase")
    axs[2].set_title("Phase Plot (omega, theta)")
    axs[2].set_xlabel("theta")
    axs[2].set_ylabel("omega")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def training_plot():
    plt.figure(figsize=(20,20))

    plt.subplot(2, 1, 1)
    plt.plot(loss_history, lw=2, color='orange', label='Loss per Epoch')
    plt.plot(np.zeros(epoch), lw=1, color='blue', label='Ideal Loss')
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(energy_history, lw=3, color='orange', label='Energy per Epoch')
    plt.plot(initial_energy * np.ones(epoch), lw=1, color='blue', label='Ideal Energy')
    plt.title("Energy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()
    
    plt.show
    
def phase_plot():   
    plt.figure(figsize=(10,10))
    plt.plot(theta_pred, omega_pred, lw=4, linestyle = "--", color='orange', label='Predicted phase')
    plt.plot(theta, omega, lw=1, color='blue', label="Exact Phase")
    plt.title("Phase Plot (omega, theta)", fontsize=20)
    plt.xlabel("theta", fontsize=20)
    plt.ylabel("omega", fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.show

def side_by_side_combined():
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0]) 
    ax2 = fig.add_subplot(gs[1, 0]) 

    ax1.plot(loss_history, lw=2, color='orange', label='Loss per Epoch')
    ax1.plot(np.zeros(epoch), lw=1, color='blue', label='Ideal Loss')
    ax1.set_title("Loss vs Epochs", fontsize=20)
    ax1.set_xlabel("Epochs", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=20)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.grid(True)
    ax1.legend(fontsize=20)

    ax2.plot(energy_history, lw=3, color='orange', label='Energy per Epoch')
    ax2.plot(initial_energy * np.ones(epoch), lw=1, color='blue', label='Ideal Energy')
    ax2.set_title("Energy vs Epochs", fontsize=20)
    ax2.set_xlabel("Epochs", fontsize=20)
    ax2.set_ylabel("Energy", fontsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.grid(True)
    ax2.legend(fontsize=20)

    ax3 = fig.add_subplot(gs[:, 1]) 

    ax3.plot(theta_pred, omega_pred, lw=4, linestyle="--", color='orange', label='Predicted Phase')
    ax3.plot(theta, omega, lw=1, color='blue', label="Exact Phase")
    ax3.set_title("Phase Plot (omega vs theta)", fontsize=20)
    ax3.set_xlabel("theta", fontsize=20)
    ax3.set_ylabel("omega", fontsize=20)
    ax3.tick_params(axis='both', labelsize=20)
    ax3.grid(True)
    ax3.legend(fontsize=20)

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------#

        # PINN for a Single Pendulum # 

#-----------------------------------------------------------------------------#

def build_model():
    model = Sequential([
        Input(shape=(1,)), # Input is time t
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(2, activation='linear', bias_initializer='random_uniform')  # Two outputs: theta and omega
    ])
    return model

#-----------------------------------------------------------------------------#

        # PILF for a Single Pendulum #

#-----------------------------------------------------------------------------#

def physics_informed_loss(t, theta_pred, omega_pred):
    # Use persistent GradientTape to compute multiple gradients
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        theta_omega = model(t)  # Predict theta and omega
        theta_pred = theta_omega[:, 0:1]
        omega_pred = theta_omega[:, 1:2]
    
    # Compute d(theta)/dt and d(omega)/dt
    dtheta_dt = tape.gradient(theta_pred, t)
    domega_dt = tape.gradient(omega_pred, t)
    
    # PIL for both ODEs:
    loss_theta = dtheta_dt - omega_pred
    loss_omega = domega_dt + tf.sin(theta_pred)

    # Use MSE to find the loss
    physics_loss = tf.reduce_mean(tf.square(loss_theta)) + tf.reduce_mean(tf.square(loss_omega))

    # Delete the tape to free memory
    del tape
    
    # Energy calculations
    energy_pred = energy_func(omega_pred, theta_pred)  # Predicted energy at each time
    energy_loss = tf.reduce_mean(tf.square(energy_pred - initial_energy))  # Penalize deviation from initial energy
    
    energy_history.append(np.average(energy_pred))

    return physics_loss + energy_loss  # Add energy loss with weight

#-----------------------------------------------------------------------------#

# Creating variables and compiling the model

#-----------------------------------------------------------------------------#

# Swing time
T = 12

# Generate training data
t_train = np.linspace(0, T, 1000).reshape(-1, 1) # Sample points from the domain [0, T]

# Convert training data to TensorFlow tensors
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Build and compile the model
model = build_model()
optimizer = Adam(learning_rate=0.0005)

# Lists for plotting
loss_history = []
energy_history = []
nrg = []
theta_history = []
omega_history = []
p_history = []

# Initial conditions
theta0 = 1
omega0 = 0

initial_energy = energy_func(omega0, theta0)


#-----------------------------------------------------------------------------#

        # Training Loop 

#-----------------------------------------------------------------------------#
epoch = 0
check = True
loss = 2

start_time = time.time()

while check == True:
    with tf.GradientTape() as tape:
        # Predict both theta and omega
        theta_omega_pred = model(t_train_tensor)
        theta_pred = theta_omega_pred[:, 0:1]
        omega_pred = theta_omega_pred[:, 1:2]

        # Physics-informed loss
        pde_loss = physics_informed_loss(t_train_tensor, theta_pred, omega_pred)
        ic_loss = tf.reduce_mean(tf.square(theta_pred[0] - theta0)) + tf.reduce_mean(tf.square(omega_pred[0] - omega0))

        # Total loss is a weighted sum of both losses
        loss = pde_loss + ic_loss

    # Compute gradients and update model weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    energy_epoch = energy_func(theta_pred, omega_pred)
    nrg.append(energy_epoch)
    theta_history.append(theta_pred)
    omega_history.append(omega_pred)
    loss_history.append(loss)
    
    # Check if the loss is < 10^-2
    if loss < 0.01:
        check = False    
    epoch = epoch + 1
    
    # Print the loss value periodically
    if epoch % 50 == 0:
        print(f"Epoch {epoch}:  Loss = {loss}:  Avg_Energy = {np.average(energy_epoch)}")
        
end_time = time.time()
               
print(f"Epoch {epoch}:  Loss = {loss}:  Avg_Energy = {np.average(energy_epoch)}")

timed = end_time - start_time
print(f"Total training time: {timed:.2f} seconds")

#-----------------------------------------------------------------------------#

        # Saving to Excel

#-----------------------------------------------------------------------------#

file_path = "training_results.xlsx"

results = pd.DataFrame({
    'Epochs': [epoch],
    'Training Time (s)': [timed]
    })

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
    results.to_excel(writer, index=False, sheet_name="Sheet1", startrow=writer.sheets['Sheet1'].max_row, header=not os.path.exists(file_path))
 
#-----------------------------------------------------------------------------#

        # Solving Numerically

#-----------------------------------------------------------------------------#    
 
def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -np.sin(theta)  
    return [dtheta_dt, domega_dt]

y0 = [(theta_pred[0].numpy())[0], (omega_pred[0].numpy())[0]]
t_span = (0, T)  
t_eval = np.linspace(*t_span, 1000)  # 1000 time points

solution = solve_ivp(pendulum, t_span, y0, t_eval=t_eval, method='RK45')

theta, omega = solution.y

#-----------------------------------------------------------------------------#

        # Plots

#-----------------------------------------------------------------------------#

fade_plot()
training_plot()
phase_plot()
side_by_side_combined()

yn = input("Output animations? (Enter y for animations) : ")

if yn == "y":

#-----------------------------------------------------------------------------#
    
        # Animations #
    
#-----------------------------------------------------------------------------#   

    #-------------------------------------------------------------------------#
    
    # Animation_1  Energy vs Time evolution
    
    #-------------------------------------------------------------------------#
    
    # Create the figure and axis for the plot
    fig1, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, np.max(t_train))  # Time range 
    ax.set_ylim(0, 1.1 * np.max(energy_history))  # Dynamic Y-axis for energy history
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs. Time')
    ax.legend()
    
    # Create a line object that will be updated
    line, = ax.plot([], [], lw=3, color='orange', label='Current Prediction')
    
    # Define the update function for the animation
    def update1(frame):
    
        # Update the plot with the energy for the current epoch
        line.set_data(t_train, nrg[frame])
    
        # Optionally update the title with the epoch number (frame)
        ax.set_title(f'Energy vs. Time (Epoch {frame + 1})')
    
        return line, ax.legend(),
    
    # Set up the animation
    ani = animation.FuncAnimation(
        fig1, update1, frames=epoch, interval=200, blit=True
    )
    
    plt.plot(initial_energy * np.ones(epoch), lw=1, color='blue', label='End Goal')
    
    # Save the animation as a GIF
    ani.save('energy_vs_time_animation.gif', writer='imagemagick', fps=50)
    plt.legend()
    plt.show()  
    
    #-------------------------------------------------------------------------#
    
    # Animation_2  Theta vs Omega evolution
    
    #-------------------------------------------------------------------------#
    
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, 1)  # Range for theta
    ax.set_ylim(-1, 1)  # Range for omega
    ax.set_xlabel('Omega')
    ax.set_ylabel('Theta')
    ax.set_title('Phase Plot (Omega vs Theta)')
    
    # Create a line object that will be updated
    line, = ax.plot([], [], lw=3, color='orange', label='Current Prediction')
    
    # Define the update function for the animation
    def update2(frame):
    
        # Update the plot with theta and omega for the current epoch
        line.set_data(theta_history[frame], omega_history[frame])
    
        # Optionally update the title with the epoch number (frame)
        ax.set_title(f'Theta vs. Omega (Epoch {frame + 1})')
    
        return line, ax.legend(),
    
    # Set up the animation
    ani = animation.FuncAnimation(
        fig2, update2, frames=epoch, interval=200, blit=True
    )
    
    plt.plot(np.cos(np.linspace(0, 2 * np.pi, 1000)), np.sin(np.linspace(0, 2 * np.pi, 1000)), lw=1, label='End Goal', color='blue')
    
    # Save the animation as a GIF
    ani.save('theta_vs_omega_animation.gif', writer='imagemagick', fps=50)
    plt.legend()
    plt.show()  # Show the final plot if needed
    
    #-------------------------------------------------------------------------#
    
    # Animation_3  X vs Z plot
    
    #-------------------------------------------------------------------------#
    
    x = np.sin(theta_history[epoch - 1])
    z = -np.cos(theta_history[epoch - 1])
    
    # Set up the figure for animation
    fig3, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-2, 0.5)
    ax.set_aspect('equal', 'box')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title('Pendulum Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    
    # Plot the initial points
    line, = ax.plot([], [], 'o-', color='orange', lw=2, label='Pendulum')
    
    # Define the initialization function for the animation
    def init():
        line.set_data([], [])
        return line, 
    
    # Define the update function for the animation
    def update3(frame):
        # Update positions for both pendulums at the current frame
        x_frame = float(x[frame + 1])
        z_frame = float(z[frame + 1])
        
        # Update the line data for the pendulums
        line.set_data([0, x_frame], [0, z_frame])  # Pendulum 
        
        return line, ax.legend(),
    
    # Create the animation
    ani = animation.FuncAnimation(fig3, update3, frames=epoch, init_func=init, blit=True, interval=200)
    
    ani.save('simple_pendulum.gif', writer='Pillow', fps=50)
    
    # Show the animation
    plt.show()
