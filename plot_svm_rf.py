import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    comparison_csv_path = "./comparison.csv"
    data = pd.read_csv(comparison_csv_path)
    print(data.head())
    sns.set_style("darkgrid", {"axes.facecolor": ".7"})
    sns.set_context(rc = {'patch.linewidth': 0.1})
    cmap = sns.light_palette("#434343") # light:b #69d ch:start=.2,rot=-.3, ch:s=.25,rot=-.25 , dark:salmon_r
    
    c = 0

    fig = plt.figure(figsize=(8,5))

    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Random Forest Accuracy')
    ax1.set_ylim(0.0, 100.0)
    ax1.set_xlabel("Data volumns (rows)")
    ax1.set_ylabel("Accuracy")

    # RF
    ax1.plot(data[data.columns[0]], data[data.columns[1]], label=data.columns[1], color=(0.0+c, 0.0+c, 0.0+c))
    ax1.legend(loc='best')
    plt.savefig("./plot/training/rf_comparison_accuracy.png", bbox_inches='tight')
    plt.clf()

    # SVM
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Random Forest Accuracy')
    ax1.set_ylim(0.0, 100.0)
    ax1.set_xlabel("Data volumns (rows)")
    ax1.set_ylabel("Accuracy")

    ax1.plot(data[data.columns[0]], data[data.columns[2]], label=data.columns[2], color=(0.0+c, 0.0+c, 0.0+c))
    ax1.legend(loc='best')
    plt.savefig("./plot/training/svm_comparison_accuracy.png", bbox_inches='tight')
    plt.clf()
    