import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #          
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  
        
                            

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = X.dot(self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        #print(f"hidden_inputs: {hidden_inputs}")
        #print(f"hidden_outputs: {hidden_outputs}")
        
        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        #print(f"- hidden_inputs: {hidden_inputs}")
        #print(f"- hidden_outputs: {hidden_outputs}")
        #print(f"- final_outputs: {final_outputs}\n")
            
        return final_outputs, hidden_outputs

    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.

        output_error_term = error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        delta_weights_i_h += hidden_error_term * X[:, None]
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
    
        return delta_weights_i_h, delta_weights_h_o    
    
    def backpropagation2(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        
               
        #### Implement the backward pass here ####
        ### Backward pass ###
        error = y - final_outputs
        output_error_term = error * hidden_outputs #* (1 - hidden_outputs)
        #print(f"output_error_term: {output_error_term}")
        hidden_error_term = np.dot(output_error_term, self.weights_hidden_to_output) * \
                   hidden_outputs * (1 - hidden_outputs)
        #delta_weights_h_o = self.lr * output_error_term * hidden_outputs
        delta_weights_h_o += hidden_outputs.T.dot(output_error_term) * self.lr
        delta_weights_i_h += hidden_error_term * X[:, None] * self.lr

        #print(f"! delta_weights_h_o: {delta_weights_h_o}")
        #print(f"! self.weights_hidden_to_output: {self.weights_hidden_to_output}")
        #print(f"! Powinna byc: [[0.07275328] [0.068271]]")
        #print(f"delta_weights_i_h: {delta_weights_i_h}")
        
        
        # TODO: Output error - Replace this value with your calculations.
        ##layer_2_error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        # TODO: Calculate the hidden layer's contribution to the error
        ##layer_2_delta = layer_2_error * hidden_outputs
        
        #print(f"- layer_2_error: {layer_2_error}")
        #print("- layer_2_delta = layer_2_error * hidden_outputs")
        #print(f"- {layer_2_delta} = {layer_2_error} * {hidden_outputs}\n\n")
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        #print(f"self.weights_input_to_hidden.T {self.weights_input_to_hidden.T}")
        ##layer_1_error = layer_2_delta.dot(self.weights_input_to_hidden.T)
        ##derivative = self.activation_function(X) * (1 - self.activation_function(X))
        ##layer_1_delta = layer_1_error * derivative
        
        #print("layer_1_delta = layer_1_error * derivative")
        #print(f"{layer_1_delta} = {layer_1_error} * {derivative}\n\n")
        #print("\n")                                                                                
                                                                                               
        # TODO: Add Weight step (input to hidden) and Weight step (hidden to output).
        # Weight step (input to hidden)
        #print(f"X.T.dot(layer_1_delta) * self.learning_rate")
        #print(f"{X.T.dot(layer_1_delta)} * {self.lr}")
        ##delta_weights_i_h += X.T.dot(layer_1_delta) * self.lr
        
        
        # Weight step (hidden to output)
        #print(f"hidden_outputs.T.dot(layer_2_delta) * self.learning_rate")
        #print(f"{hidden_outputs.T.dot(layer_2_delta)} * {self.lr}")
        ##delta_weights_h_o += hidden_outputs.T.dot(layer_2_delta) * self.lr
        #print(f"- delta_weights_h_o: {delta_weights_h_o}")
        #print(f"delta_weights_i_h & delta_weights_h_o")
        #print(f"{delta_weights_i_h} & {delta_weights_h_o}\n\n\n")
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        #print(f"1. self.weights_hidden_to_output: {self.weights_hidden_to_output}")
       
        
        # TODO: Update the weights with gradient descent step
        self.weights_hidden_to_output += delta_weights_h_o * self.lr #/ n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h * self.lr #/ n_records # update input-to-hidden weights with gradient descent step

        #print(f"2. self.weights_hidden_to_output: {self.weights_hidden_to_output}")
        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        #hidden_inputs = features.dot(self.weights_input_to_hidden) # signals into hidden layer
        #hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        #final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        #final_outputs = final_inputs # signals from final output layer 
        
        final_outputs, hidden_outputs = self.forward_pass_train(features)
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10
learning_rate = 0.7
hidden_nodes = 5
output_nodes = 1
