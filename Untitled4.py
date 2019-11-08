#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[11]:


class neuron:
    output = False
    bias_neuron = False
    activated_value = 0
    pre_func_value = 0
    local_error = 0
    def __init__(self, output, bias_neuron):
        self.output= output
        self.bias_neuron = bias_neuron
    def calc(self, value):
        self.activated_value = (np.tanh(value) if not self.output else value) if not self.bias_neuron else 1
        self.pre_func_value = value if not self.bias_neuron else 1
        return self.activated_value
    def get_error(self):
        if self.output:
            self.local_error = self.pre_func_value
        else:
            self.local_error = (1 - np.tanh(self.pre_func_value)**2)
        return self.local_error 


# In[31]:


class layer:
    neurons = 0
    if_output = False
    Neuron = []
    Weights = 0
    gradients = 0.000
    earlier_dimension = 0
    overall_output = 0
    error = 0
    earlier_output = 0
    def __init__(self, amount, if_output, earl_dim):
        self.neurons = amount
        self.if_output = if_output
        self.earlier_dimension = earl_dim
        self.Weights = np.random.rand(self.earlier_dimension, self.neurons+1)*2 -1
        self.gradients = np.zeros((self.earlier_dimension, self.neurons+1))
        self.Neuron = [neuron(output= if_output, bias_neuron = True)]
        self.create()
    def create(self):
        for i in range(self.neurons):
            self.Neuron.append(neuron(output = self.if_output, bias_neuron = False))
    def step(self, results):
        self.earlier_output = results
        output = []
        for i in range(self.neurons+1):
            output.append(self.Neuron[i].calc(np.dot(results, self.Weights[:,i])))
        self.overall_output = output
        return output
    def backstep(self):
        back_put = []
        for i in range(self.earlier_dimension):
            back_put.append(np.dot(self.overall_output, self.Weights[i,:]))
        return back_put
    def own_error(self, results):
        errors=[]
        for i in range(self.neurons+1):
            errors.append(results[i]*self.Neuron[i].get_error())
        self.error = errors
        return 0
    def update_grad(self, update):
        self.gradients = self.gradients + update*np.outer(self.earlier_output, self.error)
        return 0
    def update_weight(self, stepsize):
        self.Weights = self.Weights - stepsize * self.gradients
        return 0
            
        


# In[60]:


class MLP:
    layer = []
    layers = 0
    input_size = 0
    output = 0
    def __init__(self, layers, amount_array, input_size):
        self.layers = layers
        self.input_size = input_size
        for  i in range(layers):
            self.layer.append(layer(amount = amount_array[i],
                                    if_output = False, 
                                    earl_dim = (amount_array[i-1]+1 if i > 0 else input_size)))
        self.layer.append(layer(amount = amount_array[layers-1], if_output = True, earl_dim = amount_array[layers-2]))
    def run(self, data):
        for i in range(self.layers):
            data=self.layer[i].step(data)
        self.output = data[1:]
        return data
    def backprop(self, y_output):
        for i in reversed(range(self.layers)):
            #stimmt die Reihenfolge?
            self.layer[i].own_error(y_output)
            y_output = self.layer[i].backstep()
        return 0
    def gradient_step(self, update):
        #update = 0.5#calculated - learning #rule of quadratic error
        for i in range(self.layers):
            self.layer[i].update_grad(update)
        return 0
    def update_weights(self):
        stepsize = 1
        for i in range(self.layers):
            self.layer[i].update_weight(stepsize)
        return
    def training(self, in_label, out_label):
        approx = self.run(in_label)[0]
        error = approx - out_label
        self.backprop(self.run(in_label))
        self.gradient_step(error)
        self.update_weights()
        return 0


# In[61]:


A = MLP(3, [2,5,1], 4)
Data = [9,8,7,6]
print(A.run(Data))


# In[83]:


A.training(Data,1)


# In[84]:


A.run(Data)


# In[ ]:




