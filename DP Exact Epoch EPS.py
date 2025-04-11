# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:45:44 2025

@author: callu
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import winsound as ws

#-----------------------------------------------------------------------------#

        # Define Range #

#-----------------------------------------------------------------------------#

T = 25
it = 24*T

#-----------------------------------------------------------------------------#

        # Initializing the model #

#-----------------------------------------------------------------------------#

# family 1 :8.3433, family 2 :12.223955154418945
p = tfk.Variable(8.3433, dtype=tf.float32, trainable=True)

def energy_func(theta1, theta2, omega1, omega2):
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2 - theta1)
    s2 = np.sin(theta2 - theta1)
    energy = 3 - 2*c1 - c2*c1 +s2*s1 + omega1**2 + 1/2*omega2**2 + c2*omega1*omega2
    return(energy)

#-----------------------------------------------------------------------------#

        # Plots #
                         
#-----------------------------------------------------------------------------#

def phase_plot():
    plt.figure(figsize=(10,10))
    plt.plot(theta1_pred, omega1_pred, lw=4, color='orange', label='Predicted phase top')
    plt.plot(theta2_pred, omega2_pred, lw=4, color='chocolate', label='Predicted phase bottom')
    plt.plot(theta1, omega1, lw=1, color='blue', label="Exact Phase top")
    plt.plot(theta2, omega2, lw=1, color='navy', label="Exact Phase bottom")
    plt.title("Phase Plot (omega, theta)", fontsize=20)
    plt.xlabel("theta", fontsize=20)
    plt.ylabel("omega", fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=20)
    plt.show

def training_plot():
    plt.figure(figsize=(20,20))
    
    plt.subplot(3, 1, 1)
    plt.plot(p_history, lw=2, color='orange', label='Period per Epoch')
    plt.title("Period vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Period")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(np.log(loss_history) + 13.8, lw=2, color='orange', label='Loss per Epoch')
    plt.plot(np.zeros(epoch), lw=1, color='blue', label='Ideal Loss')
    plt.title(r"$log$(Loss) vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(r"$log$(Loss)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(energy_history, lw=3, color='orange', label='Energy per Epoch')
    plt.plot(initial_energy * np.ones(epoch), lw=1, color='blue', label='Ideal Energy')
    plt.title("Energy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()
    
    plt.show()
    
    plt.figure(figsize=(20,20))
    
    plt.subplot(3, 1, 1)
    plt.plot(p_history[-100:], lw=2, color='orange', label='Period per Epoch')
    plt.title("Final 100 Period vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Period")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(loss_history[-100:], lw=2, color='orange', label='Loss per Epoch')
    plt.plot(np.zeros(100), lw=1, color='blue', label='Ideal Loss')
    plt.title("Final 100 Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(energy_history[-100:], lw=3, color='orange', label='Energy per Epoch')
    plt.plot(initial_energy * np.ones(100), lw=1, color='blue', label='Ideal Energy')
    plt.title("Final 100 Energy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Energy")
    plt.legend()
    
    plt.show()

def three_d_phase_plot():
    fig = plt.figure(figsize=(14, 18))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(theta1_pred, omega1_pred, theta2_pred, c = theta2_pred, cmap='coolwarm')
    ax1.set_title(r'$\theta_1 \ vs \ \omega_1 \ vs \ \theta_2$', fontsize=25, y=1)
    
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(theta1_pred, omega1_pred, omega2_pred, c = omega2_pred, cmap='coolwarm')
    ax1.set_title(r'$\theta_1 \ vs \ \omega_1 \ vs \ \omega_2$', fontsize=25, y=1)
    
    return(plt.show())

def fade_plot():
    # Pendulum positions
    x1 = np.sin(theta1_pred[:,0])
    z1 = -np.cos(theta1_pred[:,0])
    x2 = np.sin(theta1_pred[:,0]) + np.sin(theta2_pred[:,0])
    z2 = -np.cos(theta1_pred[:,0]) - np.cos(theta2_pred[:,0])
    
    x1t = np.sin(theta1)
    z1t = -np.cos(theta1)
    x2t = np.sin(theta1) + np.sin(theta2)
    z2t = -np.cos(theta1) - np.cos(theta2)

    # Number of "fades"
    num_positions = 6
    indices = np.linspace(0, 80, num_positions, dtype=int)

    # Fade effect using a colormap
    colors = cm.Blues(np.linspace(0.1, 1, num_positions))
    colors2 = cm.Oranges(np.linspace(0.1, 1, num_positions))

    # Set up the figure
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    for ax in axs:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal', 'box')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Z', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    axs[0].set_title('PINN Solution', fontsize=20)
    axs[1].set_title('RK45 Solution', fontsize=20)

    # Plot multiple positions with fading effect
    for i, idx in enumerate(indices):
        axs[0].plot([0, x1[idx]], [0, z1[idx]], 'o-', color=colors[i], lw=2, alpha=0.8)
        axs[0].plot([x1[idx], x2[idx]], [z1[idx], z2[idx]], 'o-', color=colors2[i], lw=2, alpha=0.8)
        axs[1].plot([0, x1t[idx]], [0, z1t[idx]], 'o-', color=colors[i], lw=2, alpha=0.8)
        axs[1].plot([x1t[idx], x2t[idx]], [z1t[idx], z2t[idx]], 'o-', color=colors2[i], lw=2, alpha=0.8)

    plt.tight_layout()
    plt.show()

def check_plot():

    plt.figure(figsize=(20,40))
    
    plt.subplot(6, 1, 1)
    plt.plot(theta1, lw=1, color='blue', label='RK23 theta1')
    plt.plot(theta1_pred, lw=2, color='orange', label='PINN theta1')
    plt.title("theta1 vs t")
    plt.xlabel("theta1")
    plt.ylabel("t")
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(theta2, lw=1, color='blue', label='RK23 theta1')
    plt.plot(theta2_pred, lw=2, color='orange', label='PINN theta1')
    plt.title("theta2 vs t")
    plt.xlabel("theta2")
    plt.ylabel("t")
    plt.legend()
    
    plt.subplot(6, 1, 3)
    plt.plot(omega1, lw=1, color='blue', label='RK23 theta1')
    plt.plot(omega1_pred, lw=2, color='orange', label='PINN theta1')
    plt.title("omega1 vs t")
    plt.xlabel("omega1")
    plt.ylabel("t")
    plt.legend()
    
    plt.subplot(6, 1, 4)
    plt.plot(omega2, lw=1, color='blue', label='RK23 theta1')
    plt.plot(omega2_pred, lw=2, color='orange', label='PINN theta1')
    plt.title("omega2 vs t")
    plt.xlabel("omega2")
    plt.ylabel("t")
    plt.legend()   
    
    plt.plot

#-----------------------------------------------------------------------------#

        # Custom Layer for NN #

#-----------------------------------------------------------------------------#


class EPS_layer(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super(EPS_layer, self).__init__(**kwargs)

    # family 1 solns : s = (2* np.pi * inputs) / p, 
    # family 2 solns : s = (2* 8.3433 * inputs) / p
    def call(self, inputs):  
        s = (2*np.pi * inputs) / p 
        sin_out = tf.sin(s)
        cos_out = tf.cos(s)
        return tf.concat([sin_out, cos_out], axis=-1)

    def get_config(self):
        config = super(EPS_layer, self).get_config()
        return config  

#-----------------------------------------------------------------------------#

        # PINN for a Double Pendulum #

#-----------------------------------------------------------------------------#

def build_model():
    model = Sequential([
        Input(shape=(1,)), # Input is time t
        EPS_layer(),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(6, activation='linear'),
    ])
    return model


#-----------------------------------------------------------------------------#

        # PILF for double pendulum #

#-----------------------------------------------------------------------------#

def physics_informed_loss(t, theta1_pred, theta2_pred, omega1_pred, omega2_pred, tau1_pred, tau2_pred):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        X2Pendulum = model(t)  
        theta1_pred = X2Pendulum[:, 0:1]
        theta2_pred = X2Pendulum[:, 1:2]
        omega1_pred = X2Pendulum[:, 2:3]
        omega2_pred = X2Pendulum[:, 3:4]
        tau1_pred = X2Pendulum[:, 4:5]
        tau2_pred = X2Pendulum[:, 5:6]
        
    # Calculate differentials
    dtheta1_dt = tape.gradient(theta1_pred, t)
    dtheta2_dt = tape.gradient(theta2_pred, t)
    domega1_dt = tape.gradient(omega1_pred, t)
    domega2_dt = tape.gradient(omega2_pred, t)
    
    # Physics-informed loss for all ODEs 
    loss_theta1 = dtheta1_dt - omega1_pred
    loss_theta2 = dtheta2_dt - omega2_pred
    loss_omega1 = tau2_pred * tf.sin(theta2_pred - theta1_pred) - tf.sin(theta1_pred) - domega1_dt
    loss_omega2 = -tau1_pred * tf.sin(theta2_pred - theta1_pred) - domega2_dt 
    
    # Physics-informed loss for other properties
    loss_tau1 = tau1_pred - tf.cos(theta2_pred - theta1_pred) * tau2_pred - (omega1_pred) ** 2 - tf.cos(omega1_pred)
    loss_tau2 = 2 * tau2_pred - tf.cos(theta2_pred - theta1_pred) * tau1_pred - (omega2_pred) ** 2
    
    # Mean Square Error of loss terms
    mse_theta1 = tf.reduce_mean(tf.square(loss_theta1))
    mse_theta2 = tf.reduce_mean(tf.square(loss_theta2))
    mse_omega1 = tf.reduce_mean(tf.square(loss_omega1))
    mse_omega2 = tf.reduce_mean(tf.square(loss_omega2))
    mse_tau1 = tf.reduce_mean(tf.square(loss_tau1))
    mse_tau2 = tf.reduce_mean(tf.square(loss_tau2)) 
    
    physics_error = mse_omega1 + mse_omega2 + mse_theta1 + mse_theta2 + mse_tau1 + mse_tau2
    
    del tape  # Free memory
    
    # Energy calculations outside of tape
    energy_pred = energy_func(theta1_pred, theta2_pred, omega1_pred, omega2_pred)
    energy_loss = energy_pred - initial_energy
    mse_energy = tf.reduce_mean(tf.square(energy_loss))
    energy_history.append(np.average(energy_pred))

    return(physics_error + mse_energy)  # return the total error

#-----------------------------------------------------------------------------#

        # Creating variables and compiling the model

#-----------------------------------------------------------------------------#

# Create time space
t_train = np.linspace(0, T, it).reshape(-1, 1) # Sample points from the domain [0, 1]

# Convert time to TensorFlow tensors
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Build and compile the model
model = build_model()
optimizer = Adam(learning_rate=0.001)

# Use some trained weights 
#model.load_weights('model_v2.weights.h5')

# Extra lists and variables for plotting
loss_history = []
energy_history = []
p_history = []
nrg = []
theta1_history = []
theta2_history = []
omega1_history = []
omega2_history = []

# Choose initial energy
initial_energy = 0.5

#-----------------------------------------------------------------------------#

        # Training Loop #

#-----------------------------------------------------------------------------#

# Variables for training
epoch = 0
loss = 2
check = True

while check == True:
    with tf.GradientTape() as tape:
        # pull the final 6 weights from the NN for training
        theta_omega_pred = model(t_train_tensor)
        theta1_pred = theta_omega_pred[:, 0:1]
        theta2_pred = theta_omega_pred[:, 1:2]
        omega1_pred = theta_omega_pred[:, 2:3]
        omega2_pred = theta_omega_pred[:, 3:4]
        tau1_pred = theta_omega_pred[:, 4:5]
        tau2_pred = theta_omega_pred[:, 5:6]

        # Physics-informed loss
        loss = physics_informed_loss(t_train_tensor, theta1_pred, theta2_pred, omega1_pred, omega2_pred, tau1_pred, tau2_pred)
        loss = loss
        
    # Compute gradients to train the model alongside optimizer 
    gradients = tape.gradient(loss, model.trainable_variables + [p])
    optimizer.apply_gradients(zip(gradients, model.trainable_variables + [p]))

    # append apporopriate values to plot
    energy_epoch = energy_func(theta1_pred, theta2_pred, omega1_pred, omega2_pred)
    nrg.append(energy_epoch)
    theta1_history.append(theta1_pred)
    theta2_history.append(theta2_pred)
    omega1_history.append(omega1_pred)
    omega2_history.append(omega2_pred)
    p_history.append(p.numpy())
    loss_history.append(loss.numpy())
         
    # exiting criteria when training is extended     
    if epoch == 1000:
        check = False
        epoch = epoch - 1
    
    # print the results of the NN to check if the system is stuck
    if epoch % 50 == 0:
        print(f"Epoch: {epoch},     Loss: {loss},     avg_Energy: {np.average(energy_epoch)},     p_val: {p.numpy()}.")
        
    epoch = epoch + 1

#-----------------------------------------------------------------------------#

        # Solving Numerically

#-----------------------------------------------------------------------------#

# Use the initial conditions from the PINN
y0 = [
    (theta1_pred[0].numpy())[0], 
    (theta2_pred[0].numpy())[0], 
    (omega1_pred[0].numpy())[0], 
    (omega2_pred[0].numpy())[0]  
]

def double_pendulum(t, y):
    theta1, theta2, omega1, omega2 = y
    
    # solve for tau1, tau2 
    A = np.array([
        [1, -np.cos(theta2 - theta1)],
        [-0.5 * np.cos(theta2 - theta1), 1]
        ])

    b = np.array([
        omega1**2 + np.cos(omega1),
        omega2**2 / 2
        ])

    tau1, tau2 = np.linalg.solve(A, b)
    
    # Define the governing equations
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    domega1_dt = tau2 * np.sin(theta2 - theta1) - np.sin(theta1)
    domega2_dt = -tau1 * np.sin(theta2 - theta1)
    
    return [dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt]

t_eval = np.linspace(0, T, it)

# Solve using solve_ivp
solution = solve_ivp(double_pendulum, (0, T), y0, t_eval= t_eval, method='RK45')

# Extract solutions
theta1, theta2, omega1, omega2 = solution.y

#-----------------------------------------------------------------------------#

        # Plots

#-----------------------------------------------------------------------------#

phase_plot()
training_plot()
three_d_phase_plot()
fade_plot()
check_plot()

#-----------------------------------------------------------------------------#

# Alert the user that the training is finished
ws.Beep(1500, 3000)

# Determine whether to save the weights of the PINN
ny = input("Type (y) to save weights: ")
if ny == "y":
    model.save_weights('model_v2.weights.h5')

# Choose if the user wants animations
yn = input("Type (y) for animations: ")   #animations y/n

if yn == "y":

    #-----------------------------------------------------------------------------#
    
    # Animation_1  X vs Z plot
                                     
    #-----------------------------------------------------------------------------#
    
    # Pendulums 1 and 2 x and z values
    x1 = np.sin(theta1_pred[:,0])
    z1 = -np.cos(theta1_pred[:,0])
    x2 = np.sin(theta1_pred[:,0]) + np.sin(theta2_pred[:,0])
    z2 = -np.cos(theta1_pred[:,0]) - np.cos(theta2_pred[:,0])
    
    # Set up the figure for animation
    fig1, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    ax.axhline(0, color='black', linewidth=0.5)  # Ground line
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Double Pendulum Plot')
    
    # Plot the initial points
    line1, = ax.plot([], [], 'o-', color='blue', lw=2, label='Pendulum 1')
    line2, = ax.plot([], [], 'o-', color='orange', lw=2, label='Pendulum 2')
    
    # Define the initialization function for the animation
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    
    # Define the update function for the animation
    def update1(frame):
        # Update positions for both pendulums at the current frame
        x1_frame = x1[frame]
        z1_frame = z1[frame]
        x2_frame = x2[frame]
        z2_frame = z2[frame]
    
        # Update the line data for the pendulums
        line1.set_data([0, x1_frame], [0, z1_frame])  # Pendulum 1
        line2.set_data([x1_frame, x2_frame], [z1_frame, z2_frame])  # Pendulum 2
    
        return line1, line2, ax.legend()
    
    # Create the animation
    ani = animation.FuncAnimation(fig1, update1, frames=it, init_func=init, blit=True, interval=20)
    
    ani.save('DP_animation_1.gif', writer='Pillow', fps= 24)
    
    # Show the animation
    plt.show()
    
    #-----------------------------------------------------------------------------#
    
    # Animation_2  Energy vs Time evolution
    
    #-----------------------------------------------------------------------------#
    
    # Create the figure and axis for the plot
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, np.max(t_train))  # Time range 
    ax.set_ylim(initial_energy*0.8, initial_energy*1.25)  # Dynamic Y-axis for energy history
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs. Time')
    ax.legend()
    
    # Create a line object that will be updated
    line, = ax.plot([], [], lw=3, color = 'blue', label='Current Prediction')
    
    # Define the update function for the animation
    def update2(frame):
    
        # Update the plot with the energy for the current epoch
        line.set_data(t_train, nrg[frame])
    
        # Optionally update the title with the epoch number (frame)
        ax.set_title(f'Energy vs. Time (Epoch {frame + 1})')
    
        return line, ax.legend(),
    
    if epoch < 1000:
        ani = animation.FuncAnimation(
            fig2, update2, frames = epoch, interval = 200, blit=True)
    else:
        ani = animation.FuncAnimation(
            fig2, update2, frames=range(0, len(nrg), 10), interval = 200, blit=True)
    
    plt.plot(initial_energy*np.ones(epoch), lw = 1, color = 'orange', label='End Goal')
    
    # Save the animation as a GIF
    ani.save('DP_animation_2.gif', writer='imagemagick', fps=50)
    plt.legend()
    plt.show()  
    
    #-----------------------------------------------------------------------------#
    
    # Animation_3  Theta vs Omega evolution
    
    #-----------------------------------------------------------------------------#
    
    
    fig3, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.4, 0.4)  # Range for theta
    ax.set_ylim(-0.4, 0.4)  # Range for omega
    ax.set_xlabel('Omega')
    ax.set_ylabel('Theta')
    ax.set_title('Phase Plot (Omega vs Theta)')
    
    # Create a line object that will be updated
    line1, = ax.plot([], [], lw=1, color = 'blue', label='Pendulum 1 Prediction')
    line2, = ax.plot([], [], lw=1, color = 'orange', label='Pendulum 2 Prediction')
    
    # Define the update function for the animation
    def update3(frame):
    
        # Update the plot with theta and omega for the current epoch
        line1.set_data(theta1_history[frame], omega1_history[frame])
        line2.set_data(theta2_history[frame], omega2_history[frame])
    
        # Optionally update the title with the epoch number (frame)
        ax.set_title(f'Theta vs. Omega (Epoch {frame + 1})')
    
        return line1, line2, ax.legend(),
    
    if epoch < 1000:
        ani = animation.FuncAnimation(
            fig3, update3, frames = epoch, interval = 200, blit=True)
    else:
        ani = animation.FuncAnimation(
            fig3, update3, frames=range(0, len(theta1_history), 10), interval = 200, blit=True)
    
    # Save the animation as a GIF
    ani.save('DP_animation_3.gif', writer='imagemagick', fps=10)
    plt.legend()
    plt.show()

