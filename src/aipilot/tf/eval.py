import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import itertools
from pckgLogger import logger
from sklearn.metrics import auc, roc_curve

class Evaluator(object):
    def __init__(self, model, train_generator, val_generator, task='classification'):
        self.logger = logger
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.class_names = list(self.train_generator.class_indices.keys())
        self.target_labels = self.val_generator.labels
        predictions = model.predict(self.val_generator, verbose=True)
        self.predictions = np.argmax(predictions, axis=1)
        self.task = task

    def confusion_matrix(self, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if self.task == 'classification':
            cm = confusion_matrix( self.target_labels, self.predictions )
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(self.class_names))
            plt.xticks(tick_marks, range(len(self.class_names)), rotation=45)
            plt.yticks(tick_marks, range(len(self.class_names)))
            
            target_names = self.class_names

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            return plt 
        else:
            self.logger.error(f'Confusion Matrix is NOT available for "{self.task}" task 🚫')
    
    def classification_report(self):
        if self.task == 'classification':
            print(classification_report(self.predictions, self.target_labels,target_names=self.class_names))
        else:
            self.logger.error(f'Classification Report is NOT available for "{self.task}" task 🚫')
    
    def auc_roc(self):
        if self.task == 'classification':
            fpr, tpr, thresholds = roc_curve(self.target_labels, self.predictions)
            auc_value = auc(fpr, tpr)
            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='AUC (area = {:.3f})'.format(auc_value))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()
            return plt
        else:
            self.logger.error(f'AUC ROC analysis is NOT available for "{self.task}" task 🚫')

