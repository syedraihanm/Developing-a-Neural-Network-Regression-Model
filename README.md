# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.
<img width="1095" height="745" alt="542714748-b2141eb2-ad6c-4f3d-8419-c8126c9732b7" src="https://github.com/user-attachments/assets/a10c57a4-bf02-4043-9a23-10da995e3d1c" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Syed Mohamed Raihan M

### Register Number: 212224240167

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')


```

### Dataset Information
<img width="159" height="142" alt="image" src="https://github.com/user-attachments/assets/0fa8fab5-a78d-4a68-bfaf-a3651c25070e" />



### OUTPUT
<img width="375" height="221" alt="image" src="https://github.com/user-attachments/assets/39e514ea-e199-45c8-96cf-e09c219a086d" />

<img width="216" height="33" alt="image" src="https://github.com/user-attachments/assets/b7518663-d4f2-4d0f-b988-8ccc8aeef845" />


### Training Loss Vs Iteration Plot
<img width="721" height="559" alt="image" src="https://github.com/user-attachments/assets/b5febbe6-805a-40e0-83c3-68f27198e23c" />


### New Sample Data Prediction
<img width="300" height="25" alt="image" src="https://github.com/user-attachments/assets/d4be3069-84fa-47e5-b105-ba9e4dc88132" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
