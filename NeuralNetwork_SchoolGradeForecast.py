from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

dataSet = SupervisedDataSet(2,1)

#Exemplos para treino -> Tempo de sono, tempo de estudo e o terceiro parametro é a nota
dataSet.addSample((0.8, 0.5), (0.75))
dataSet.addSample((0.4, 0.6), (0.5))
dataSet.addSample((0.9, 0.7), (0.85))
dataSet.addSample((0.3, 0.2), (0.3))
dataSet.addSample((1.0, 0.9), (0.95))
dataSet.addSample((0.9, 0.2), (0.4))

#Arquitetura da Rede Neural
neuralNetwork = buildNetwork(2, 4, 1, bias=True) # 2 neuronios de entrada, 4 neuronios na camada oculta e um neuronio na camada de saída

trainer = BackpropTrainer(neuralNetwork, dataSet)

for i in range(2000):
    print(trainer.train()) #Treinando o sistema

while True:
    sleep = float(input('Tempo de sono: '))
    study = float(input('Tempo de estudo: '))

    x = neuralNetwork.activate((sleep, study)) [0] * 10 #Computando dados

    print(f'A previsão da nota é {x}')