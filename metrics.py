import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score,accuracy_score,recall_score, f1_score
import seaborn as sns




def confMat(test, pred, cl_name, name):
    # Plot non-normalized confusion matrix
    mat = metrics.confusion_matrix(test, pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(mat,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=cl_name,
                yticklabels=cl_name,
                annot=True,
                fmt='d')
    plt.title(name)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def val(test, pred):
    accuracy = accuracy_score(test, pred)
    precision = precision_score(test, pred, average='macro')
    recall = recall_score(test, pred, average='macro')
    f1 = f1_score(test, pred, average='macro')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1 score:", f1)
    print("\n\n\n")