#!/bin/python
#-----------------------------------------------------------------------------
# File Name : brain2_perceptron_learn.py
# Author: Emre Neftci
#
# Creation Date : Thu 06 Oct 2016 05:10:03 PM PDT
# Last Modified : Thu 06 Oct 2016 11:48:17 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from brian2 import *
from npamlib import *

def bin_rate(r_mon, duration):
    '''
    Calculates the mean rate of PopulationRateMonitor r_mon
    '''
    return np.array(r_mon.rate).reshape(-1, duration/defaultclock.dt).mean(axis=1)

n_samples = 100

# Learn the random points
LEARN_MODE_MNIST = False
# Learn the digits 8 or 3
LEARN_MODE_MNIST = True

#Neuron Parameters (modify these)
Cm = 0.05*pF; gl = 1e-9*siemens; taus = 20*ms
Vt = 10*mV; Vr = 0*mV;

#How long each stimulus was presented
duration = 200*ms

eqs = '''
dv/dt  = - gl*v/Cm 
         + isyn/Cm + Vt*gl/Cm: volt (unless refractory)
disyn/dt  = -isyn/taus : amp 
'''

data, labels = ann_createDataSet(n_samples) #create 20 2d data samples
blabels = (labels+1)//2 #labels -1,1 to 0,1
data = (1+data)/2 #inputs in the range 0,1

#Load digits 3 and 8 only

if LEARN_MODE_MNIST:
    data, labels = data_load_mnist([3,8])
    data = data[:n_samples,:]
    labels = labels[:n_samples]==3
    print "Using perceptron learning rule on MNIST digits..."
else:
    print "Using perceptron learning rule on random coordinates..."

    
print "Below is the PROPORTION correct for just the learning rule, not on spiking neurons..."

#convert labels to True / False
labelsTF = (labels==labels[0])
#Train a data sample with trained perceptron:
w2, res = ann_train_perceptron(data[:100], labelsTF[:100], n = 1000, eta = .1)

wbias = w2[0]
wdata = w2[1:]

print "STATUS: Bias & Data Set"

## Spiking Network
#Following 2 lines for time-dependent inputs
rate = TimedArray(data*100*Hz, dt = duration)
Pdata = NeuronGroup(data.shape[1], 'rates = rate(t,i) : Hz', threshold='rand()<rates*dt')


print "STATUS: Rate & Neuron Group Set"


#Input bias
Pbias = PoissonGroup(1, rates = 100*Hz)
P = NeuronGroup(1, eqs, threshold='v>Vt', reset='v = Vr',
                refractory=20*ms, method='milstein')

print "STATUS: Bias Set"

Sdata = Synapses(Pdata, P, 'w : amp', on_pre='isyn += w')
Sdata.connect() #Connect all-to-all
Sdata.w = wdata*nA

Sbias = Synapses(Pbias, P, 'w : amp', on_pre='isyn += w')
Sbias.connect() #Connect all-to-all
Sbias.w = wbias*nA

print "STATUS: Synapses Set"

s_mon = SpikeMonitor(P)
r_mon = PopulationRateMonitor(P) #Monitor spike rates
s_mon_data = SpikeMonitor(Pdata)

print "STATUS: Monitors Set"

run(n_samples*duration)
print "STATUS: Simulation Finished"

output_rate = bin_rate(r_mon, duration)

decision_rule = 50

if LEARN_MODE_MNIST:
    decision_rule = 46

output_rate_over_threshold = output_rate >= decision_rule

output_matched_label = output_rate_over_threshold == labelsTF
number_matched_label = sum(output_matched_label)
percentage_matched_label = 100 * (number_matched_label / float(output_matched_label.size))

#print output_rate
#stim_show(data)
#print(labels)
print "Spiking Number Correct:" , number_matched_label 
print "Spiking Percent Correct:" , percentage_matched_label , "%"
