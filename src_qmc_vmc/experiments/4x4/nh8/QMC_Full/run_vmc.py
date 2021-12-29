import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ============= Functions and Classes

class VariationalMonteCarlo(tf.keras.Model):

    # Constructor
    def __init__(self, Lx, Ly, 
                 V, Omega, delta,
                 num_hidden, learning_rate,
                 trunc=2, seed=1234):
        
        super(VariationalMonteCarlo, self).__init__()

        """ PARAMETERS """
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.V        = V               # Van der Waals potential
        self.Omega    = Omega           # Rabi frequency
        self.delta    = delta           # Detuning
        self.trunc    = trunc           # Truncation, set to Lx+Ly for none, default is 2

        self.N        = Lx * Ly         # Number of spins
        self.nh       = num_hidden      # Number of hidden units in the RNN
        self.seed     = seed            # Seed of random number generator 
        self.K        = 2               # Dimension of the local Hilbert space

        # Set the seed of the rng
        tf.random.set_seed(self.seed)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate, epsilon=1e-8)

        # Build the model RNN
        # RNN layer: N -> nh
        self.rnn = tf.keras.layers.GRU(self.nh, kernel_initializer='glorot_uniform',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.001),
                                       return_sequences = True,
                                       return_state = True,
                                       stateful = False)

        # Dense layer: nh - > K
        self.dense = tf.keras.layers.Dense(self.K, activation = tf.nn.softmax,
                                           kernel_regularizer = tf.keras.regularizers.l2(0.001))

        # Generate the list of bonds for NN,NNN,NNNN on a 
        # square lattice with open boundaries
        self.buildlattice()
    
    @tf.function
    def sample(self,nsamples):
        # Zero initialization for visible and hidden state 
        inputs = 0.0*tf.one_hot(tf.zeros(shape=[nsamples,1],dtype=tf.int32),depth=self.K)
        hidden_state = tf.zeros(shape=[nsamples,self.nh])

        logP = tf.zeros(shape=[nsamples,],dtype=tf.float32)

        for j in range(self.N):
            # Run a single RNN cell
            rnn_output,hidden_state = self.rnn(inputs,initial_state=hidden_state)
            # Compute log probabilities
            probs = self.dense(rnn_output)
            log_probs = tf.reshape(tf.math.log(1e-10+probs),[nsamples,self.K])
            # Sample
            sample = tf.random.categorical(log_probs,num_samples=1)
            if (j == 0):
                samples = tf.identity(sample)
            else:
                samples = tf.concat([samples,sample],axis=1)
            # Feed result to the next cell
            inputs = tf.one_hot(sample,depth=self.K)
            add = tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)

            logP = logP+tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)

        return samples,logP

    @tf.function
    def logpsi(self,samples):
        # Shift data
        num_samples = tf.shape(samples)[0]
        data = tf.one_hot(samples[:,0:self.N-1],depth=self.K)

        x0 = 0.0*tf.one_hot(tf.zeros(shape=[num_samples,1],dtype=tf.int32),depth=self.K)
        inputs = tf.concat([x0,data],axis=1)
        
        hidden_state = tf.zeros(shape=[num_samples,self.nh])
        rnn_output,_ = self.rnn(inputs,initial_state = hidden_state)
        probs        = self.dense(rnn_output)
            
        log_probs   = tf.reduce_sum(tf.multiply(tf.math.log(1e-10+probs),tf.one_hot(samples,depth=self.K)),axis=2)
        
        return 0.5 * tf.reduce_sum(log_probs, axis=1)

    #@tf.function
    def localenergy(self,samples,logpsi):
        eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)

        # Chemical potential
        for j in range(self.N):
            eloc += - self.delta * tf.cast(samples[:,j],tf.float32)
     
        for n in range(len(self.nns)):
            eloc += (self.V/self.nns[n][0]) * tf.cast(samples[:,self.nns[n][1]]*samples[:,self.nns[n][2]],tf.float32)

        flip_logpsi = tf.zeros(shape=[tf.shape(samples)[0]])

        # Off-diagonal part
        for j in range(self.N):
            flip_samples = np.copy(samples)
            flip_samples[:,j] = 1 - flip_samples[:,j]
            flip_logpsi = self.logpsi(flip_samples)
            eloc += -0.5*self.Omega * tf.math.exp(flip_logpsi-logpsi)
            
        return eloc

    """ Generate the square lattice structures """
    def coord_to_site(self,x,y):
        return self.Ly*x+y
    
    def buildlattice(self):
        self.nns = []
        
        for n in range(1,self.Lx):
            for n_ in range(n+1):
                
                if n+n_ > self.trunc:
                    continue
        
                else:
                    for x in range(self.Lx-n_):
                        for y in range(self.Ly-n):
                            coeff = np.sqrt(n**2+n_**2)**6
                            if n_ == 0 :
                                self.nns.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x,y+n)])
                            elif n == n_: 
                                self.nns.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y+n)])
                                self.nns.append([coeff,self.coord_to_site(x+n,y),self.coord_to_site(x,y+n)])
                            else:
                                self.nns.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n_,y+n)])
                                self.nns.append([coeff,self.coord_to_site(x+n_,y),self.coord_to_site(x,y+n)])
                            
                    for y in range(self.Ly-n_):
                        for x in range(self.Lx-n):
                            coeff = np.sqrt(n**2+n_**2)**6
                            if n_ == 0 :
                                self.nns.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y)])
                            elif n == n_: 
                                continue #already counted above
                            else:
                                self.nns.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y+n_)])
                                self.nns.append([coeff,self.coord_to_site(x,y+n_),self.coord_to_site(x+n,y)])


def create_tf_dataset(uploaded_files, data_step_size=100):
    '''
    create tensor flow data set from uploaded files
    data_step_size (int): determines step size when loading data
    '''
    data = []
    for file in uploaded_files:
        new_data = uploaded_files[file]
        new_data = new_data.astype(int)
        new_data = new_data[::data_step_size]
        #print("New data shape: ", np.array(new_data).shape)
        data.extend(new_data)

    #convert to tf.data.Dataset
    data = np.array(data)
    print("Overall dataset shape: ", data.shape) #shape = (Num_examples, N)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset

def run_VMC(vmc, epochs, delta, qmc_data, energy, variance, batch_size=100):
    '''
    Run RNN using vmc sampling or qmc data. If qmc_data is None, uses vmc sampling. 
    Otherwise uses qmc data loaded in qmc_data
    '''

    if qmc_data != None:
        print("Running VMC using QMC data for delta = ", delta)
    else:
        print("Running VMC for delta =",delta)

    # You can remove the tqdm() here to get rid of the status bar
    for n in range(1, epochs+1):
        #for n in tqdm(range(1,epochs+1)):
        
        #use qmc_data to update RNN weights
        if qmc_data != None:
            dset = qmc_data.shuffle(len(qmc_data))
            dset = dset.batch(batch_size)
        
            for i, batch in enumerate(dset):
                # Evaluate the loss function in AD mode
                with tf.GradientTape() as tape:
                    logpsi = vmc.logpsi(batch)
                    
                    loss = - 2.0 * tf.reduce_mean(logpsi)

                # Compute the gradients either with qmc_loss
                gradients = tape.gradient(loss, vmc.trainable_variables)
              
                # Update the parameters
                vmc.optimizer.apply_gradients(zip(gradients, vmc.trainable_variables))

        else:
            samples, _ = vmc.sample(ns)
      
            # Evaluate the loss function in AD mode
            with tf.GradientTape() as tape:
                sample_logpsi = vmc.logpsi(samples)
                with tape.stop_recording():
                    sample_eloc = tf.stop_gradient(vmc.localenergy(samples, sample_logpsi))
                    sample_Eo = tf.stop_gradient(tf.reduce_mean(sample_eloc))
                  
                sample_loss = tf.reduce_mean(2.0*tf.multiply(sample_logpsi, tf.stop_gradient(sample_eloc)) - 2.0*sample_Eo*sample_logpsi)
          
                # Compute the gradients either with sample_loss
                gradients = tape.gradient(sample_loss, vmc.trainable_variables)
        
                # Update the parameters
                vmc.optimizer.apply_gradients(zip(gradients, vmc.trainable_variables))
           
        #append the energy to see convergence
        samples, _ = vmc.sample(ns)
        sample_logpsi = vmc.logpsi(samples)
        sample_eloc = vmc.localenergy(samples, sample_logpsi)

        energies = sample_eloc.numpy()
        avg_E = np.mean(energies)/float(N)
        var_E = np.var(energies)/float(N)
        energy.append(avg_E) #average over samples

    return vmc, energy, variance

# ============= Main program

import os
import glob

# Hamiltonian parameters
Lx = 4     # Linear size in x direction
Ly = 4     # Linear size in y direction
N = Lx*Ly   # Total number of atoms:w
V = 7.0     # Strength of Van der Waals interaction
Omega = 1.0 # Rabi frequency
delta = 1.0 # Detuning

# RNN-VMC parameters
lr = 0.001     # learning rate of Adam optimizer
nh = 8        # Number of hidden units in the GRU cell
ns = 1000     # Number of samples used to approximate the energy at each step
qmc_epochs = 4000 # Training iterations for qmc, if 0 only do vmc
vmc_epochs = 0 # Training iterations for vmc, if 0 only do qmc
total_epochs = vmc_epochs+qmc_epochs # Total training iterations
seed = 1234    # Seed of RNG
batch_size = 100 # Batch size for QMC training
skip_data = 100 # Skip elements in QMC data set

exact_energy = {4: -0.4534132086591546, 8: -0.40518005298872917, 12:-0.3884864748124427 , 16: -0.380514770608724}

wavefunction = VariationalMonteCarlo(Lx,Ly,V,Omega,delta,nh,lr,Lx+Ly,seed)
energy = []
variance = []

if qmc_epochs != 0:

    path = "../../../../../QMC_data"
    dim_path = "Dim={}_M=1000000_V={}_omega={}_delta={}".format(Lx, int(V), Omega, delta) # Can change this to look at Dim = 4, 8, 12, 16
    files_we_want = glob.glob(os.path.join(path,dim_path,'samples*'))
    uploaded = {}
    for file in files_we_want:
        data = np.loadtxt(file)
        uploaded[file] = data

    qmc_dataset = create_tf_dataset(uploaded, data_step_size=skip_data) 
 
    # Optimize with data first
    wavefunction, energy, variance = run_VMC(wavefunction, qmc_epochs, delta, qmc_dataset, energy, variance, batch_size)

    #a_file = open("E12qmc_lr001.dat", "w")
    #np.savetxt(a_file, energy)
    #a_file.close()

    #wavefunction.save_weights("qmc.weights")
    #wavefunction.load_weights("qmc.weights")

if vmc_epochs != 0:
    wavefunction, energy, variance = run_VMC(wavefunction, vmc_epochs, delta, None, energy, variance, batch_size)
    
a_file = open("E_QMC_{}_{}.dat".format(Lx,nh), "w")
np.savetxt(a_file, energy)
a_file.close()
