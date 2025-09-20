
import matplotlib.pyplot as plt
from OpenSetRecognition import Evaluation

class AnaliseGrafica:

    def __init__(self, nome:str,nome_dataset:str):
        self.nome=nome
        self.nome_dataset=nome_dataset
        self.accuracy=[]
        self.inner_metric=[]
        self.outer_metric=[]
        self.halfpoint=[]
        self.uuc_accuracy=[]
        self.F1=[]
        self.epochs=[]

    def addEpoch(self,metricas:Evaluation,epoch:int):
        self.epochs.append(epoch)

        self.accuracy.append(metricas.accuracy)
        self.inner_metric.append(metricas.inner_metric)
        self.outer_metric.append(metricas.outer_metric)
        self.halfpoint.append(metricas.halfpoint)
        self.uuc_accuracy.append(metricas.uuc_accuracy)
        #falta o F1, preciso ver como esta em Evaluation

        print(f"{self.nome} inner metric is %.3f ({metricas.certas_inner}/{metricas.total_inner})" % (metricas.inner_metric))
        print(f"{self.nome} outer metric is %.3f ({metricas.certas_outer}/{metricas.total_outer})" % (metricas.outer_metric))
        print(f"{self.nome} halfpoint is %.3f ({metricas.certas_halfpoint}/{metricas.total_halfpoint})" % (metricas.halfpoint))
        print(f"{self.nome} uuc accuracy is %.3f ({metricas.certas_uuc_accuracy}/{metricas.total_ucc_accuracy})" % (metricas.uuc_accuracy))
        print(f"{self.nome} accuracy is %.3f" % (metricas.accuracy))
        print(f"{self.nome} F1 is %.3f" % (metricas.f1_measure))
        print(f"{self.nome} f1_macro is %.3f" % (metricas.f1_macro))
        print(f"{self.nome} f1_macro_weighted is %.3f" % (metricas.f1_macro_weighted))
        print(f"{self.nome} area_under_roc is %.3f" % (metricas.area_under_roc))
        print(f"_________________________________________")
    
    def mostraGrafico(self):
        plt.plot(self.epochs, self.accuracy, color='red', label='Acurácia')
        plt.plot(self.epochs, self.inner_metric, color='blue', label='Inner metric')
        plt.plot(self.epochs, self.outer_metric, color='orange', label='Outer metric')
        plt.plot(self.epochs, self.halfpoint, color='green', label='Halfpoint')
        
        titulo = f"metricas do {self.nome} "
        
            
        plt.title(titulo)

        plt.xlabel("Épocas")
        plt.xticks(self.epochs)
        plt.ylabel("Valor da Métrica")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(f"../../../metricas_{self.nome}.png")
        plt.show()

        