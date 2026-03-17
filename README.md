# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.
 
## THEORY

Problem Statement

Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships. A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.


## Neural Network Model
<img width="1115" height="695" alt="547549011-75c8b8af-fd98-4c8f-91f8-6249d4585e78" src="https://github.com/user-attachments/assets/ddf2c9bf-5821-45a2-9ac8-eaeab11eb11a" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

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

## PROGRAM
### Name: KUSHMA S
### Register Number:212224040168
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df1=pd.read_csv("/content/nn-dl-exp.csv")
X = df1[['input']].values
y = df1[['output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,10)
        self.fc2=nn.Linear(10,18)
        self.fc3=nn.Linear(18,1)
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
    loss=criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch%200==0:
      print(f'Epoch [{epoch}/{epochs}], Loss:{loss.item():.6f}')

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_n1_1 = torch.tensor([[3]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
     
```
## Dataset Information

<img width="277" height="517" alt="564202285-2e6f6033-ea51-4154-9e57-226bec0ccbec" src="https://github.com/user-attachments/assets/ab979600-e939-495b-9872-d75fc8d7c6b5" />

## OUTPUT

<img width="767" height="523" alt="564202923-13c5f7e6-161e-4e66-b02e-977f80657b58" src="https://github.com/user-attachments/assets/4123621c-a862-407e-845b-10d0c6b91d4e" />

### New Sample Data Prediction
<img width="831" height="273" alt="564203188-c1f28d56-85ff-47e5-a3f7-c87a37cdf3ea" src="https://github.com/user-attachments/assets/947fe4f9-1de7-48b2-ac4a-69a621d650d3" />

## RESULT
The neural network regression model was successfully developed and trained.
