import time as t2
import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt  
import seaborn as sns  
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest  # Classical model
import pandas as pd  # For handling CSV input
# Step 1: Load the dataset from a custom CSV file
file_path = input("Please enter the path to your CSV file: ")
# Starting the time measurement
start_time = t2.time()
try:
    # Load the CSV file
    df = pd.read_csv(file_path)
    # Assuming the CSV has 'time', 'amount', and 'label' columns
    time = df['time'].values
    amount = df['amount'].values
    label = df['label'].values
except FileNotFoundError:
    print("The specified file was not found. Please check the path and try again.")
    exit()
except pd.errors.EmptyDataError:
    print("The provided CSV file is empty.")
    exit()
except pd.errors.ParserError:
    print("Error parsing the CSV file. Please check the file format.")
    exit()
# Combine into a dataset
data = np.column_stack((time, amount, label))
# Separate the features and labels
X = data[:, :2]  # First two columns are features (time and amount)
y = data[:, 2]   # Third column is the label (0 or 1)
# Step 2: Normalize the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
#Time Stamp 1
T_1=t2.time()
# Step 4: Amplitude Encoding
def amplitude_encoding(data):
    norm_data = data / np.linalg.norm(data)
    qml.templates.AmplitudeEmbedding(norm_data, wires=[0, 1], pad_with=0, normalize=True)
# Step 5: Build Quantum Fourier Transform
def qft(wires):
    qml.Hadamard(wires=wires[0])
    for i in range(1, len(wires)):
        for j in range(i):
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(np.pi / 2**(i - j), wires=wires[i])
    for i in range(len(wires) // 2):
        qml.SWAP(wires=[wires[i], wires[len(wires) - i - 1]])
# Step 6: Build Hybrid QNN Model
dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    amplitude_encoding(inputs)
    qml.RY(weights[0], wires=0)  # Example of a parameterized rotation
    qft(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))  # Return expectation values
def build_qnn_model(input_shape):
    weight_shapes = {"weights": (1,)}  # Shape for weights
    model = keras.Sequential()
    model.add(qml.qnn.KerasLayer(quantum_circuit, 
                                  output_dim=2,  # Output dimensions match the QNode output
                                  input_shape=input_shape, 
                                  weight_shapes=weight_shapes))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    return model
# Define input shape and build model
input_shape = (2,)  # Two features
model = build_qnn_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Step 7: Train the Quantum Model
quantum_start_time = t2.time()  # Start time for the quantum model
model.fit(X_train, y_train, epochs=5, batch_size=4, validation_split=0.2)
quantum_end_time = t2.time()  # End time for the quantum model
quantum_execution_time = quantum_end_time - quantum_start_time  # Execution time for the quantum model
# Step 8: Make Predictions
predictions = model.predict(X_test)
predicted_labels_qnn = (predictions > 0.5).astype(int)
# Step 9: Evaluate the Quantum Model
print("Quantum Model Classification Report:")
print(classification_report(y_test, predicted_labels_qnn))
conf_matrix_qnn = confusion_matrix(y_test, predicted_labels_qnn)
print("Quantum Model Confusion Matrix:")
print(conf_matrix_qnn)
#End Quantum Model
E_Q=t2.time()
# Step 10: Evaluate Classical Model (Isolation Forest)
classical_start_time = t2.time()  # Start time for the classical model
isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
isolation_forest.fit(X_train)
y_pred_classical = isolation_forest.predict(X_test)
y_pred_binary_classical = [1 if x == -1 else 0 for x in y_pred_classical]
classical_end_time = t2.time()  # End time for the classical model
classical_execution_time = classical_end_time - classical_start_time  # Execution time for the classical model
# Print classical model results
print("Classical Model Classification Report:")
print(classification_report(y_test, y_pred_binary_classical))
conf_matrix_classical = confusion_matrix(y_test, y_pred_binary_classical)
print("Classical Model Confusion Matrix:")
print(conf_matrix_classical)
# Step 11: Calculate and print total execution time
end_time = t2.time()
total_execution_time = end_time - start_time
print(f"The time taken to execute the code is {total_execution_time:.2f} seconds")
print(f"Quantum Model Execution Time: {(E_Q-start_time):.2f} seconds")
print(f"Classical Model Execution Time: {(T_1-start_time)+(end_time-E_Q):.2f} seconds")
# Step 12: Visualizations
# Confusion Matrix for Quantum Model
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_qnn, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Quantum Model Confusion Matrix')
# Confusion Matrix for Classical Model
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_classical, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Classical Model Confusion Matrix')
plt.tight_layout()
plt.show()
# Step 13: ROC Curve for Quantum Model
fpr_qnn, tpr_qnn, _ = roc_curve(y_test, predictions)  # Using raw predictions for ROC
roc_auc_qnn = auc(fpr_qnn, tpr_qnn)
plt.figure(figsize=(8, 6))
plt.plot(fpr_qnn, tpr_qnn, color='blue', label=f'Quantum Model ROC Curve (area = {roc_auc_qnn:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Quantum Model Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()
# Step 14: ROC Curve for Classical Model
fpr_classical, tpr_classical, _ = roc_curve(y_test, y_pred_binary_classical)
roc_auc_classical = auc(fpr_classical, tpr_classical)
plt.figure(figsize=(8, 6))
plt.plot(fpr_classical, tpr_classical, color='green', label=f'Classical Model ROC Curve (area = {roc_auc_classical:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Classical Model Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()