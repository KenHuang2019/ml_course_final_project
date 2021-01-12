import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    lstm_root_path = "./plot/training/lstm/hist_joblib/"

    sns.set_style("darkgrid", {"axes.facecolor": ".7"})
    sns.set_context(rc = {'patch.linewidth': 0.1})
    cmap = sns.light_palette("#434343") # light:b #69d ch:start=.2,rot=-.3, ch:s=.25,rot=-.25 , dark:salmon_r
    
    c = 0

    fig = plt.figure(figsize=(8,5))

    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('LSTM Accuracy')
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation accuracy")

    # ax2 = fig.add_subplot(1,2,2)
    # ax2.set_title('LSTM Loss')
    # ax2.set_xlabel("Epochs")
    # ax2.set_ylabel("Validation loss")

    sorted_dir = sorted(os.listdir(lstm_root_path), reverse=True)
    sorted_dir.append(sorted_dir[0])
    sorted_dir.pop(0)

    for h in sorted_dir:
        hist_path = lstm_root_path + h
        print(hist_path)
        hist = joblib.load(hist_path)
        rows = h.split("_")[0]
        # ax1.plot(hist['accuracy'], label='Train accuracy', color=(0.0+c, 0.0+c, 0.0+c))
        ax1.plot(hist['val_accuracy'], label=rows + ' rows', color=(0.0+c, 0.0+c, 0.0+c))
        print("Max acc: ", round(max(hist['val_accuracy']), 2))
        # ax2.plot(hist['loss'], label='Train loss', color=(0.0+c, 0.0+c, 0.0+c))
        # ax2.plot(hist['val_loss'], label=rows + ' rows', color=(0.0+c, 0.0+c, 0.0+c))

        c += 0.2
    
    ax1.legend(loc='best')
    # ax2.legend(loc='best')
    plt.savefig("./plot/training/comparison/lstm_comparison_accuracy.png", bbox_inches='tight')
    plt.clf()
    
    print("##################################################################################")
    attention_root_path = "./plot/training/attention/hist_joblib/"

    c = 0

    fig = plt.figure(figsize=(8,5))

    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Transformer Accuracy')
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation accuracy")

    # ax2 = fig.add_subplot(1,2,2)
    # ax2.set_title('Transformer Loss')
    # ax2.set_xlabel("Epochs")
    # ax2.set_ylabel("Validation loss")

    sorted_dir = sorted(os.listdir(attention_root_path), reverse=True)
    sorted_dir.append(sorted_dir[0])
    sorted_dir.pop(0)

    for h in sorted_dir:
        hist_path = attention_root_path + h
        print(hist_path)
        hist = joblib.load(hist_path)
        rows = h.split("_")[0]
        # ax1.plot(hist['accuracy'], label='Train accuracy', color=(0.0+c, 0.0+c, 0.0+c))
        ax1.plot(hist['val_accuracy'], label=rows + ' rows', color=(0.0+c, 0.0+c, 0.0+c))
        print("Max acc: ", round(max(hist['val_accuracy']), 2))
        
        # ax2.plot(hist['loss'], label='Train loss', color=(0.0+c, 0.0+c, 0.0+c))
        # ax2.plot(hist['val_loss'], label=rows + ' rows', color=(0.0+c, 0.0+c, 0.0+c))

        c += 0.2
    
    ax1.legend(loc='best')
    # ax2.legend(loc='best')
    plt.savefig("./plot/training/comparison/attention_comparison_accuracy.png", bbox_inches='tight')
    plt.clf()
    print("##################################################################################")
    ann_root_path = "./plot/training/ann/hist_joblib/"
    c = 0

    fig = plt.figure(figsize=(8,5))

    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('ANN Accuracy')
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation accuracy")

    sorted_dir = sorted(os.listdir(ann_root_path), reverse=True)
    sorted_dir.append(sorted_dir[0])
    sorted_dir.pop(0)
    
    for h in sorted_dir:
        hist_path = ann_root_path + h
        print(hist_path)
        hist = joblib.load(hist_path)
        rows = h.split("_")[0]
        # ax1.plot(hist['accuracy'], label='Train accuracy', color=(0.0+c, 0.0+c, 0.0+c))
        ax1.plot(hist['val_accuracy'], label=rows + ' rows', color=(0.0+c, 0.0+c, 0.0+c))
        print("Max acc: ", round(max(hist['val_accuracy']), 2))
        
        # ax2.plot(hist['loss'], label='Train loss', color=(0.0+c, 0.0+c, 0.0+c))
        # ax2.plot(hist['val_loss'], label=rows + ' rows', color=(0.0+c, 0.0+c, 0.0+c))

        c += 0.2
    
    ax1.legend(loc='best')
    # ax2.legend(loc='best')
    plt.savefig("./plot/training/comparison/ann_comparison_accuracy.png", bbox_inches='tight')
    plt.clf()

