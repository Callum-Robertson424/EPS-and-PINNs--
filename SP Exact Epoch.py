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

def energy_func(theta, omega):
    energy = 0.5 * (omega * omega) + 1 - np.cos(theta)
    return(energy)

def plot1():
    # Find the exact solution of the Simple Pendulum
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

def plot2():
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

#-----------------------------------------------------------------------------#

# Neural Network with 1 input and 2 outputs 

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

# Loss Function for a Simple Pendulum 

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
    
    # Physics-informed loss for both ODEs:
    loss_theta = dtheta_dt/T - omega_pred
    loss_omega = domega_dt/T + tf.sin(theta_pred)

    # Combine the losses from both equations using MSE
    physics_loss = tf.reduce_mean(tf.square(loss_theta)) + tf.reduce_mean(tf.square(loss_omega))

    # Delete the tape to free memory
    del tape
    
    # Energy conservation penalty
    energy_pred = energy_func(theta_pred, omega_pred)  # Predicted energy at each time
    energy_loss = tf.reduce_mean(tf.square(energy_pred - initial_energy))  # Penalize deviation from initial energy
    
    energy_history.append(np.average(energy_epoch))

    return physics_loss + energy_loss  # Add energy loss with weight

#-----------------------------------------------------------------------------#

# Creating variables and compiling the model

#-----------------------------------------------------------------------------#

# Generate training data
t_train = np.linspace(0, 1, 1000).reshape(-1, 1) # Sample points from the domain [0, 10]

# Convert training data to TensorFlow tensors
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Build and compile the model
model = build_model()
optimizer = Adam(learning_rate=0.001)

# Initial conditions
theta0 = 1
omega0 = 0
T=10

# Lists for plotting
loss_history = []
energy_history = []
energy_epoch = 0
nrg = []
theta_history = []
omega_history = []

initial_energy = energy_func(theta0, omega0)

#-----------------------------------------------------------------------------#

# Training Loop 

#-----------------------------------------------------------------------------#

epoch = 400

start_time = time.time()

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        # Predict both theta and omega
        theta_omega_pred = model(t_train_tensor)
        theta_pred = theta_omega_pred[:, 0:1]
        omega_pred = theta_omega_pred[:, 1:2]

        # Physics-informed loss & IC loss
        pde_loss = physics_informed_loss(t_train_tensor, theta_pred, omega_pred)
        ic_loss = tf.reduce_mean(tf.square(theta_pred[0] - theta0)) + tf.reduce_mean(tf.square(omega_pred[0] - omega0))

        # Total loss 
        loss = pde_loss + ic_loss

    # Compute gradients and update model weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    energy_epoch = energy_func(theta_pred, omega_pred)
    nrg.append(energy_epoch)
    theta_history.append(theta_pred)
    omega_history.append(omega_pred)
    loss_history.append(loss.numpy())
    
    # Print the loss value periodically
    if epoch % 50 == 0:
        print(f"Epoch {epoch}:  Loss = {loss}:  Avg_Energy = {np.average(energy_epoch)}")
        
end_time = time.time()
 
print(f"Epoch {epoch}:  Loss = {loss}:  Avg_Energy = {np.average(energy_epoch)}")
timed = end_time - start_time
print(f"Total training time: {timed:.2f} seconds")

# Saving to Excel

file_path = "training_results.xlsx"

results = pd.DataFrame({
    'Loss': [loss.numpy()],
    'Training Time (s)': [timed]
    })

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
    results.to_excel(writer, index=False, sheet_name="Sheet1", startrow=writer.sheets['Sheet1'].max_row, header=not os.path.exists(file_path))
    
#-----------------------------------------------------------------------------#

# Plots

#-----------------------------------------------------------------------------#

plot1()
plot2()

#-----------------------------------------------------------------------------#
    
# Animations
    
#-----------------------------------------------------------------------------#

#yn = input("Output animations? (Enter y) : ")
yn = "n"

if yn == "y":

    #-----------------------------------------------------------------------------#
    
    # Animation_1  Energy vs Time evolution
    
    #-----------------------------------------------------------------------------#
    
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
    
    #-----------------------------------------------------------------------------#
    
    # Animation_2  Theta vs Omega evolution
    
    #-----------------------------------------------------------------------------#
    
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
    
    #-----------------------------------------------------------------------------#
    
    # Animation_3  X vs Z plot
    
    #-----------------------------------------------------------------------------#
    
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
