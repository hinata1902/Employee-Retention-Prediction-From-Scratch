# Project Summary:
# The goal of this task is to build a neural network model that predicts whether an employee will leave the present company or not depending on various factors. Given a dataset containing historical employee information, the model should predict whether a current employee is likely to leave the company.
# The dataset contains the following Columns:
# (a) Education: The highest level of education attained by the individual.
# (b) JoiningYear: The year in which the employee joined the current company.
# (c) City: The city that the individual belongs to.
# (d) PaymentTier: Depending on their current salary, the individuals are classified into different tiers.
# (e) Age: Age of the employee.
# (f) Gender: The gender of the individual.
# (g) EverBenched: This is a boolean which tells whether the employee was ever put on the bench in the current company.
# (h) ExperienceInCurrentDomain: Years of experience the individual has in the domain they are currently working.
# (i) LeaveOrNot: This is the target variable (boolean) which tells whether the employee will leave the company or not.
# In this project, no pretrained models were used. The neural network model was built from scratch.


import pandas 
import torch
import numpy 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def data_preprocessing(task_1a_dataframe):
	task_1a_dataframe = task_1a_dataframe.drop(['EverBenched'], axis=1)
	LabelEncoder = preprocessing.LabelEncoder()
	task_1a_dataframe['Education'] = LabelEncoder.fit_transform(task_1a_dataframe['Education'])    
	task_1a_dataframe['City'] = LabelEncoder.fit_transform(task_1a_dataframe['City'])
	task_1a_dataframe['Gender'] = LabelEncoder.fit_transform(task_1a_dataframe['Gender'])
	scaler = StandardScaler()
	task_1a_dataframe[['JoiningYear', 'Age', 'PaymentTier', 'ExperienceInCurrentDomain']] = scaler.fit_transform(task_1a_dataframe[['JoiningYear', 'Age', 'PaymentTier', 'ExperienceInCurrentDomain']])
	encoded_dataframe = task_1a_dataframe
	return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
	x = encoded_dataframe.drop(columns=['LeaveOrNot'], axis=1)
	y = encoded_dataframe['LeaveOrNot']
	features_and_targets = [x, y]
	return features_and_targets

def load_as_tensors(features_and_targets):
	X = features_and_targets[0]
	y = features_and_targets[1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
	scalar = StandardScaler()
	x_train = scalar.fit_transform(X_train)
	x_test = scalar.transform(X_test)
	X_train = torch.tensor(X_train.values, dtype=torch.float32)
	y_train = torch.tensor(y_train.values, dtype=torch.float32)
	X_test = torch.tensor(X_test.values, dtype=torch.float32)
	y_test = torch.tensor(y_test.values, dtype=torch.float32)
	y_train = y_train.view(y_train.shape[0], 1)
	y_test = y_test.view(y_test.shape[0], 1)
	tensors_and_iterable_training_data = [X_train, X_test, y_train, y_test]
	return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		self.fc1 = nn.Linear(7, 128)
		self.fc2 = nn.Linear(128, 32)
		self.fc3 = nn.Linear(32, 1)

	def forward(self, y_predicted):
		y_predicted = torch.relu(self.fc1(y_predicted))
		y_predicted = torch.relu(self.fc2(y_predicted))
		y_predicted = torch.sigmoid(self.fc3(y_predicted))
		predicted_output = y_predicted
		return predicted_output

def model_loss_function():
	loss_function = nn.BCELoss()
	return loss_function

def model_optimizer(model):
	learning_rate = 0.001  
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	return optimizer

def model_number_of_epochs():
	number_of_epochs = 500
	return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	model.train()
	for epoch in range(number_of_epochs):
		y_predicted1 = model(tensors_and_iterable_training_data[0])
		loss = loss_function(y_predicted1, tensors_and_iterable_training_data[2])
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
	model.eval()
	with torch.no_grad():
		y_predicted = model(tensors_and_iterable_training_data[1])
		y_predicted_cls = y_predicted.round()
	trained_model = y_predicted_cls		
	return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
	model_accuracy = accuracy_score(trained_model, tensors_and_iterable_training_data[3])
	model_accuracy = model_accuracy * 100
	return model_accuracy

if __name__ == "__main__":
	task_1a_dataframe = pandas.read_csv("Employee-Retention-Prediction-From-Scratch\dataset.csv")
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	model = Salary_Predictor()
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)
	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")
	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "Employee-Retention-Prediction-From-Scratch/model")

