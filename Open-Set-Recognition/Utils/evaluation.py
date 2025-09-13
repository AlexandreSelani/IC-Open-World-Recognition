import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


class Evaluation(object):
    """Evaluation class based on python list"""
    def __init__(self, predict, label,prediction_scores = None, unknown_class_idx=None):
        self.predict = predict
        self.label = label
        self.prediction_scores = prediction_scores
        self.unknown_class_idx = unknown_class_idx # indice que representa a classe desconhecida. Vai ser usado para calcular as metricas inner e outer

        self.inner_metric, self.certas_inner, self.total_inner = self._inner_metric()
        self.uuc_accuracy, self.certas_uuc_accuracy, self.total_ucc_accuracy = self._UUC_Accuracy()
        self.accuracy,self.certas_accuracy,self.total_accuracy = self._accuracy()
        self.outer_metric, self.certas_outer, self.total_outer = self._outer_metric()
        self.f1_measure = self._f1_measure()
        self.f1_macro = self._f1_macro()
        self.f1_macro_weighted = self._f1_macro_weighted()
        self.precision, self.recall = self._precision_recall(average='micro')
        self.precision_macro, self.recall_macro = self._precision_recall(average='macro')
        self.precision_weighted, self.recall_weighted = self._precision_recall(average='weighted')
        self.confusion_matrix = self._confusion_matrix()
        if self.prediction_scores is not None:
            self.area_under_roc = self._area_under_roc(prediction_scores)

    def _accuracy(self) -> tuple[float,int,int]:
        """
        Returns the accuracy score of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict)),correct,len(self.predict)
    
    def _inner_metric(self) -> tuple[float,float,float]:
        """Retorna a acuracia levando em consideracao apenas as amostras de classes CONHECIDAS (Inner metric ou KKC Accuracy)"""
        assert len(self.predict) == len(self.label)
        
        indices_amostras = [i for i,x in enumerate(self.label) if x != self.unknown_class_idx] #vetor com os indices das amostras que devem ser verificadas
        predicoes = [self.predict[i] for i in indices_amostras] #amostras a serem consideradas

        assert len(indices_amostras) == len(predicoes)

        corretas = 0

        for predicao, idx in zip(predicoes,indices_amostras):
            if predicao == self.label[idx]: #se a predicao for correta
                corretas+=1

        return float(corretas)/float(len(predicoes)),float(corretas),float(len(predicoes))

    def _UUC_Accuracy(self) -> tuple[float,int,int]:
        """Retorna a acuracia levando em consideracao apenas as amostras de classes DESCONHECIDAS (UUC Accuracy)
        NAO eh outer metric
        """

        assert len(self.predict) == len(self.label)
        
        indices_amostras = [i for i,x in enumerate(self.label) if x == self.unknown_class_idx] #vetor com os indices das amostras que devem ser verificadas
        predicoes = [self.predict[i] for i in indices_amostras] #amostras a serem consideradas

        assert len(indices_amostras) == len(predicoes)

        corretas = 0

        for predicao, idx in zip(predicoes,indices_amostras):
            if predicao == self.label[idx]: #se a predicao for correta
                corretas+=1

        return float(corretas)/float(len(predicoes)),corretas,len(predicoes)
    
    def _outer_metric(self) -> tuple[float,int,int]:
        """Mede a habilidade do classificador de distinguir KKCs e UUCs. Eh um problema de classificacao binaria
        """
        assert len(self.predict) == len(self.label)
        corretas = 0

        for predicao,label_correta in zip(self.predict,self.label):
            if(label_correta == self.unknown_class_idx):#se a amostra for UUC
                if(predicao==self.unknown_class_idx):#se o classificador detectou a novidade
                    corretas+=1
            else:                                   #se a amostra for KKC
                if(predicao!=self.unknown_class_idx): #se a amostra foi classificada como KKC, independente de acertar a classe
                    corretas+=1
        
        return float(corretas)/float(len(self.predict)),corretas,len(self.predict)

    
    def _f1_measure(self) -> float:
        """
        Returns the F1-measure with a micro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='micro')

    def _f1_macro(self) -> float:
        """
        Returns the F1-measure with a macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro')

    def _f1_macro_weighted(self) -> float:
        """
        Returns the F1-measure with a weighted macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='weighted')

    def _precision_recall(self, average) -> (float, float):
        """
        Returns the precision and recall scores for the label and predictions. Observes the average type.

        :param average: string, [None (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
            For explanations of each type of average see the documentation for
            `sklearn.metrics.precision_recall_fscore_support`
        :return: float, float: representing the precision and recall scores respectively
        """
        assert len(self.predict) == len(self.label)
        precision, recall, _, _ = precision_recall_fscore_support(self.label, self.predict, average=average)
        return precision, recall

    def _area_under_roc(self, prediction_scores: np.array = None, multi_class='ovo') -> float:
        """
        Area Under Receiver Operating Characteristic Curve

        :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires
            prediction scores for each class. If not specified, will generate its own prediction scores that assume
            100% confidence in selected prediction.
        :param multi_class: {'ovo', 'ovr'}, default='ovo'
            'ovo' computes the average AUC of all possible pairwise combinations of classes.
            'ovr' Computes the AUC of each class against the rest.
        :return: float representing the area under the ROC curve
        """
        label, predict = self.label, self.predict
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        # assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores, prediction_scores, multi_class=multi_class)

    def _confusion_matrix(self, normalize=None) -> np.array:
        """
        Returns the confusion matrix corresponding to the labels and predictions.

        :param normalize: {‘true’, ‘pred’, ‘all’}, default=None.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        :return:
        """
        assert len(self.predict) == len(self.label)
        return confusion_matrix(self.label, self.predict, normalize=normalize)

    def plot_confusion_matrix(self, labels: [str] = None, normalize=None, ax=None, savepath=None) -> None:
        """

        :param labels: [str]: label names
        :param normalize: {‘true’, ‘pred’, ‘all’}, default=None.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        :param ax: matplotlib.pyplot axes to draw the confusion matrix on. Will generate new figure/axes if None.
        :return:
        """
        conf_matrix = self._confusion_matrix(normalize)  # Evaluate the confusion matrix
        display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)  # Generate the confusion matrix display

        # Formatting for the plot
        if labels:
            xticks_rotation = 'vertical'
        else:
            xticks_rotation = 'horizontal'

        display.plot(include_values=True, cmap=plt.colormaps.get_cmap('Blues'), xticks_rotation=xticks_rotation, ax=ax)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, bbox_inches='tight', dpi=200)
        plt.close()


if __name__ == '__main__':
    predict = [1, 2, 3, 4, 5, 3, 3, 2, 2, 5, 6, 6, 4, 3, 2, 4, 5, 6, 6, 3, 2]
    label =   [2, 5, 3, 4, 5, 3, 2, 2, 4, 6, 6, 6, 3, 3, 2, 5, 5, 6, 6, 3, 3]

    eval = Evaluation(predict, label,unknown_class_idx=6)
    print(f"Inner metric: {eval.inner_metric}%")
    print(f"Outer metric: {eval.outer_metric}%")
    print('Accuracy:', f"%.3f" % eval.accuracy)
    print('F1-measure:', f'{eval.f1_measure:.3f}')
    print('F1-macro:', f'{eval.f1_macro:.3f}')
    print('F1-macro (weighted):', f'{eval.f1_macro_weighted:.3f}')
    print('precision:', f'{eval.precision:.3f}')
    print('precision (macro):', f'{eval.precision_macro:.3f}')
    print('precision (weighted):', f'{eval.precision_weighted:.3f}')
    print('recall:', f'{eval.recall:.3f}')
    print('recall (macro):', f'{eval.recall_macro:.3f}')
    print('recall (weighted):', f'{eval.recall_weighted:.3f}')

    # Generate "random prediction score" to test feeding in prediction score from NN
    test_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    test_one_hot_encoder.fit(np.array(label).reshape(-1, 1))
    rand_prediction_scores = 2 * test_one_hot_encoder.transform(np.array(predict).reshape(-1, 1))  # One hot
    rand_prediction_scores += np.random.rand(*rand_prediction_scores.shape)
    # rand_prediction_scores /= rand_prediction_scores.sum(axis=1)[:, None]
    # print('Area under ROC curve (with 100% confidence in prediction):', f'{eval.area_under_roc():.3f}')
    # print('Area under ROC curve (variable probability across classes):',
    #       f'{eval.area_under_roc(prediction_scores=rand_prediction_scores):.3f}')
    # print(eval.confusion_matrix)
    label_names = ["bird","bog","perople","horse","cat", "unknown"]
    eval.plot_confusion_matrix(normalize="true",labels=label_names)

    print(classification_report(label, predict, digits=3))
