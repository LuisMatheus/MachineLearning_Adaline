import numpy as np

class Adaline:
    def __init__(self, input_values , output_values , learning_rate , activation_function , precision):
        oneColumns = np.ones((len(input_values), 1 )) * -1
        self.input_values = np.append(oneColumns,input_values,axis=1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(self.input_values.shape[1])
        self.precision = precision
        
        self.epochs = 0
        self.err = []
        
    def errQuadMed(self):
        EQM = 0
        for x,d in zip(self.input_values,self.output_values):
            
            u = np.dot(x,self.W)

                        
            EQM = EQM + pow((d - u),2)
            
        return EQM/len(self.output_values)
    
    def train(self):
        e = 1     #erro absoluto
        
        while e > self.precision :
 
            e_old = self.errQuadMed()
                                    
            for x,d in zip(self.input_values,self.output_values):
                
                u =  np.dot(x,self.W)
                
                self.W = self.W + (self.learning_rate * (d - u) * x)
                
            e_new = self.errQuadMed()
            
            self.epochs +=1
            
            e = np.abs(e_new - e_old)
            
            self.err.append(e)
            
        print(self.W)
        print(self.epochs)
        
    def evaluate(self,input_value):
        u = np.dot(input_value,self.W)
        return self.activation_function.g(u)