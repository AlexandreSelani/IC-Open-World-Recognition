
import matplotlib.pyplot as plt
from OpenSetRecognition import Evaluation
from AnaliseGrafica import AnaliseGrafica

class AnaliseGrafica_OpenMax(AnaliseGrafica):

    def __init__(self,nome_dataset:str):
        super().__init__("OpenMax",nome_dataset)
    
    def mostraGrafico(self,tail=None,alpha=None,epsilon=None,batch_size=None):
        plt.plot(self.epochs, self.accuracy, color='red', label='Acurácia')
        plt.plot(self.epochs, self.inner_metric, color='blue', label='Inner metric')
        plt.plot(self.epochs, self.outer_metric, color='orange', label='Outer metric')
        plt.plot(self.epochs, self.halfpoint, color='green', label='Halfpoint')
        
        titulo = f"metricas do {self.nome} (tail = {tail}, alpha = {alpha}, epsilon = {epsilon}, batch_size = {batch_size}) = {self.nome_dataset}"
        
            
        plt.title(titulo)

        plt.xlabel("Épocas")
        plt.xticks(self.epochs)
        plt.ylabel("Valor da Métrica")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(f"../../../metricas_openmax_{self.nome_dataset}.png")
        plt.show()

        