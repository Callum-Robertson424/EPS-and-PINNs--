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
from tensorflow.keras.initializers import Initializer
import tensorflow.keras.backend as K


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import winsound as ws

#-----------------------------------------------------------------------------#

                                # Take inputs #

#-----------------------------------------------------------------------------#

T = 25
it = 24*T

#-----------------------------------------------------------------------------#

                            # Initializing the model #

#-----------------------------------------------------------------------------#

# Trainable period variable
n = 2       # set n =/= 1, for EnPS
p = tfk.Variable(n*8.344828605651855, dtype=tf.float32, trainable=True)   

def energy_func(theta1, theta2, omega1, omega2):
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2 - theta1)
    s2 = np.sin(theta2 - theta1)
    energy = 3 - 2*c1 - c2*c1 +s2*s1 + omega1**2 + 1/2*omega2**2 + c2*omega1*omega2
    return(energy)

#-----------------------------------------------------------------------------#

                           # Parameters for training #

#-----------------------------------------------------------------------------#

initial_energy = 0.5
final_energy = 0.5
err = 0.0000001

#-----------------------------------------------------------------------------#

                         # Plots to pull from training #
                         
#-----------------------------------------------------------------------------#

def plot2(p_history, loss_history, epoch, energy_history, initial_energy):

    
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
    
    return(plt.show())

def plot3(p_history, loss_history, epoch, energy_history, initial_energy):    
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
    
    return(plt.show())

def plot4(theta1_pred, theta2_pred, omega1_pred, omega2_pred):
    fig = plt.figure(figsize=(14, 18))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(theta1_pred, omega1_pred, theta2_pred, c = theta2_pred, cmap='coolwarm')
    ax1.set_title(r'$\theta_1 \ vs \ \omega_1 \ vs \ \theta_2$', fontsize=25, y=1)
    
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(theta1_pred, omega1_pred, omega2_pred, c = omega2_pred, cmap='coolwarm')
    ax1.set_title(r'$\theta_1 \ vs \ \omega_1 \ vs \ \omega_2$', fontsize=25, y=1)
    
    return(plt.show())

def plot5(theta1_pred, theta2_pred, omega1_pred, omega2_pred):
    # Pendulum positions
    x1 = np.sin(theta1_pred[:,0])
    z1 = -np.cos(theta1_pred[:,0])
    x2 = np.sin(theta1_pred[:,0]) + np.sin(theta2_pred[:,0])
    z2 = -np.cos(theta1_pred[:,0]) - np.cos(theta2_pred[:,0])

    #number of "fades"
    num_positions = 6
    indices = np.linspace(0, 80, num_positions, dtype=int)  # Select 8 evenly spaced frames within the first 300

    # Fade effect using a colormap
    colors = cm.Blues(np.linspace(0.1, 1, num_positions))  # Blue shades for pendulum 1
    colors2 = cm.Oranges(np.linspace(0.1, 1, num_positions))  # Orange shades for pendulum 2

    # Set up the figure
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-3, 3)
    axs[0].set_aspect('equal', 'box')
    axs[0].axhline(0, color='black', linewidth=0.5)
    axs[0].axvline(0, color='black', linewidth=0.5)
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Z')
    axs[0].set_title('Double Pendulum Fade Plot')

    # Plot multiple positions with fading effect
    for i, idx in enumerate(indices):
        axs[0].plot([0, x1[idx]], [0, z1[idx]], 'o-', color=colors[i], lw=2, alpha=0.8)  # Pendulum 1
        axs[0].plot([x1[idx], x2[idx]], [z1[idx], z2[idx]], 'o-', color=colors2[i], lw=2, alpha=0.8)  # Pendulum 2
        
    axs[1].plot(theta1_pred, omega1_pred, color = 'blue')
    axs[1].plot(theta2_pred, omega2_pred, color = 'orange')
    axs[1].set_title("Phase Plot")
    axs[1].set_xlabel("theta")
    axs[1].set_ylabel("omega")
    axs[1].set_xlim(-0.7, 0.7)
    axs[1].set_ylim(-0.7, 0.7)
    axs[1].set_aspect('equal', 'box')
        
    plt.show()

#-----------------------------------------------------------------------------#

# Custom Layer for NN 

#-----------------------------------------------------------------------------#

class e_p_sol(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super(e_p_sol, self).__init__(**kwargs)

    def call(self, inputs):  
        s = (2 * np.pi * inputs) / p # Compute sin and cos of the input
        sin_out = tf.sin(s)
        cos_out = tf.cos(s)
        return tf.concat([sin_out, cos_out], axis=-1)

    def get_config(self):
        config = super(e_p_sol, self).get_config()
        return config  

#-----------------------------------------------------------------------------#

# Custom Initialization for NN 

#-----------------------------------------------------------------------------#

M = 0       # mean
SD = 2      # standard deviation
MV = 1    # max value
mv = -MV    # min value

class initialized_points(Initializer):
    def __init__(self, mean = M, stddev = SD, minval = mv, maxval = MV):
        self.mean = mean
        self.stddev = stddev
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape, dtype=None):
        values = K.random_normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
        return tf.clip_by_value(values, self.minval, self.maxval)  

#-----------------------------------------------------------------------------#

                         # PINN for double pendulum #

#-----------------------------------------------------------------------------#

def build_model():
    circular_init = initialized_points(mean = M, stddev = SD, minval = mv, maxval = MV)
    model = Sequential([
        Input(shape=(1,)), # Input is time t
        e_p_sol(),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(100, activation='swish', bias_initializer='lecun_uniform'),
        Dense(6, activation='linear',
              kernel_initializer=circular_init,   # Custom init for circle
              bias_initializer=circular_init) , 
    ])
    return model

                              # Loss Function #

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

    dtheta1_dt = tape.gradient(theta1_pred, t)
    dtheta2_dt = tape.gradient(theta2_pred, t)
    domega1_dt = tape.gradient(omega1_pred, t)
    domega2_dt = tape.gradient(omega2_pred, t)
    
                   # Physics-informed loss for all ODEs # 
    
    loss_theta1 = dtheta1_dt - omega1_pred
    loss_theta2 = dtheta2_dt - omega2_pred
    loss_omega1 = tau2_pred * tf.sin(theta2_pred - theta1_pred) - tf.sin(theta1_pred) - domega1_dt
    loss_omega2 = -tau1_pred * tf.sin(theta2_pred - theta1_pred) - domega2_dt 
    
                   # Physics-informed loss for tension properties #
    
    loss_tau1 = tau1_pred - tf.cos(theta2_pred - theta1_pred) * tau2_pred - (omega1_pred) ** 2 - tf.cos(omega1_pred)
    loss_tau2 = 2 * tau2_pred - tf.cos(theta2_pred - theta1_pred) * tau1_pred - (omega2_pred) ** 2
    
    avg_theta1 = tf.reduce_mean(tf.square(loss_theta1))
    avg_theta2 = tf.reduce_mean(tf.square(loss_theta2))
    avg_omega1 = tf.reduce_mean(tf.square(loss_omega1))
    avg_omega2 = tf.reduce_mean(tf.square(loss_omega2))
    avg_tau1 = tf.reduce_mean(tf.square(loss_tau1))
    avg_tau2 = tf.reduce_mean(tf.square(loss_tau2)) 
     
    physics_loss = avg_omega1 + avg_omega2 + avg_theta1 + avg_theta2 + avg_tau1 + avg_tau2

    del tape  # Delete the tape to free memory
    
    # Energy conservation penalty
    energy_pred = energy_func(theta1_pred, theta2_pred, omega1_pred, omega2_pred)  # Predicted energy at each time
    energy_loss = tf.reduce_mean(tf.square(energy_pred - initial_energy))  # Penalize deviation from initial energy
    
    energy_history.append(np.average(energy_pred))

    return(physics_loss + 0.8*energy_loss)  # Add energy loss with weight

    
    return(plt.show)

#-----------------------------------------------------------------------------#

            # Creating the time vector and compiling the model #

#-----------------------------------------------------------------------------#

# Create time space
t_train = np.linspace(0, T, it).reshape(-1, 1) # Sample points from the domain [0, 1]

# Convert time to TensorFlow tensors
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Build and compile the model
model = build_model()
optimizer = Adam(learning_rate=0.001)
model.load_weights('model.weights.h5')

# Extra lists and variables for plotting
loss_history = []
energy_history = []
p_history = []
nrg = []
theta1_history = []
theta2_history = []
omega1_history = []
omega2_history = []

#-----------------------------------------------------------------------------#

                         # Create the training loop #

#-----------------------------------------------------------------------------#
epoch = 0
tepoch = 0
prev_epoch = 0
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
        #ic_loss = tf.reduce_mean(tf.square(theta1_pred[0] - theta1_0)) + tf.reduce_mean(tf.square(theta2_pred[0] - theta2_0)) + tf.reduce_mean(tf.square(omega1_pred[0] - omega1_0)) + tf.reduce_mean(tf.square(omega2_pred[0] - omega2_0)) 
        loss = loss #+ ic_loss
        
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
    if epoch == 10000:
        check = False
        epoch = epoch - 1
    
    # print the results of the NN to check if the system is stuck
    if epoch % 50 == 0:
        print(f"Epoch: {epoch},     Loss: {loss},     avg_Energy: {np.average(energy_epoch)},     p_val: {p.numpy()}.")
        
    epoch = epoch + 1
    tepoch = tepoch + 1

#-----------------------------------------------------------------------------#

                        # Plotting the entire training

#-----------------------------------------------------------------------------#

plot5(theta1_pred, theta2_pred, omega1_pred, omega2_pred)
plot2(p_history, loss_history, epoch, energy_history, initial_energy)
plot3(p_history, loss_history, epoch, energy_history, initial_energy)
plot4(theta1_pred, theta2_pred, omega1_pred, omega2_pred)

#-----------------------------------------------------------------------------#

# Alert the user that the training is finished
ws.Beep(1500, 3000)

# Determine whether to save the weights of the PINN
ny = input("Type (y) to save weights: ")
if ny == "y":
    model.save_weights('model.weights.h5')

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
    
    if tepoch < 1000:
        ani = animation.FuncAnimation(
            fig2, update2, frames = tepoch, interval = 200, blit=True)
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
    
    if tepoch < 1000:
        ani = animation.FuncAnimation(
            fig3, update3, frames = tepoch, interval = 200, blit=True)
    else:
        ani = animation.FuncAnimation(
            fig3, update3, frames=range(0, len(theta1_history), 10), interval = 200, blit=True)
    
    # Save the animation as a GIF
    ani.save('DP_animation_3.gif', writer='imagemagick', fps=10)
    plt.legend()
    plt.show()

