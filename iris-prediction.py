from NeuralNetwork import *
from random import randint, shuffle
'''
reference: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names

Summary Statistics:
	             Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826
   sepal width : 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
   petal width : 0.1  2.5   1.20  0.76    0.9565  (high!)
'''
def tanh(x):
	return math.tanh(x)

def dtanh(y):
	return 1 - (y * y)

class Iris:

	def __init__(self):
		self.nn = NeuralNetwork([4,4,3],0.1,tanh,dtanh)

	def training(self, filename):
		aux = []
		file = open(filename)
		for line in file:	aux.append(line.strip('\n'))
		file.close()
		for i in range(0,50):
			self.epoch(aux)

	def epoch(self, data_set_array):
		shuffle(data_set_array)
		aux = []
		inputs = []
		for i in range(0,len(data_set_array)):
			aux = data_set_array[i].split(',')
			if aux[4] == "Iris-versicolor":
				target = [0,1,0]
			elif aux[4] == "Iris-setosa":
				target = [1,0,0]
			elif aux[4] == "Iris-virginica":
				target = [0,0,1]
			inputs.append(float(aux[0])/7.9)
			inputs.append(float(aux[1])/4.4)
			inputs.append(float(aux[2])/6.9)
			inputs.append(float(aux[3])/2.5)
			self.nn.train(inputs, target)
			del inputs[:]

	def predict(self, array_input):
		#feedfoward
		'''
		returns 1 for setosa
		returns 2 for versicolor
		returns 3 for virginica
		'''
		choice = self.nn.feedfoward(array_input)
		# print(choice)
		if max(choice) == choice[0]:
			return 1
		elif max(choice) == choice[1]:
			return 2
		elif max(choice) == choice[2]:
			return 3


if __name__ == "__main__":
	'''
	file pattern:
	4 floats as inputs and the target as a string in the end, all separeted by
	commas
	'''
	test = Iris()
	test.training("iris-train-data-set")
	aux = []
	aux2 = []
	inputs = []
	guess = []
	correct=0
	file = open("iris-test-data-set")
	for line in file:	aux.append(line.strip('\n'))
	file_length = len(aux)
	file.close()
	shuffle(aux)
	for i in range(0,len(aux)):
		aux2 = aux[i].split(',')
		inputs.append(float(aux2[0])/7.9)
		inputs.append(float(aux2[1])/4.4)
		inputs.append(float(aux2[2])/6.9)
		inputs.append(float(aux2[3])/2.5)
		choice = test.predict(inputs)
		if choice == 1:
			guess = "Iris-setosa"
		elif choice == 2:
			guess = "Iris-versicolor"
		elif choice == 3:
			guess = "Iris-virginica"
		if guess == aux2[4]:
			correct+=1
		# print(guess + " " + aux2[4])
		del aux2[:]
		del inputs[:]
	print("Percentage: "+str((correct/file_length)))
