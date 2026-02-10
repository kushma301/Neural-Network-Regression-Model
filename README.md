# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.
 
## THEORY

Problem Statement

Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships. A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.


## Neural Network Model

<img width="869" height="713" alt="546204037-bec82200-a092-45c8-a3f1-f988e7eac296" src="https://github.com/user-attachments/assets/af2d39c4-34fb-47a7-a51c-c5fc955890f1" />

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
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(nethraa_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = nethraa_brain(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

```
## Dataset Information

<img width="196" height="351" alt="546198423-81525732-5cf7-433b-9b70-56cd7cf89865" src="https://github.com/user-attachments/assets/e5099449-a17c-4ff8-996a-35a3baead34c" />


## OUTPUT

<img width="686" height="555" alt="546203150-a4845d31-00fe-48da-b18f-7d15bd9459c4" src="https://github.com/user-attachments/assets/080e2a13-091d-448e-9877-f2aec3a58398" />


### New Sample Data Prediction
<img width="793" height="400" alt="546199086-5d566fda-2ea2-4352-a96f-d9b8c625496b" src="https://github.com/user-attachments/assets/5ed38d7a-5708-47fc-98fb-ec6438c0f984" />


## RESULT
The neural network regression model was successfully developed and trained.
