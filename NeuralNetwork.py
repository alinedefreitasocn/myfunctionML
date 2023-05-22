import matplotlib.pyplot as plt


def plot_history(history):
    """
    Ploting the Loss and the accuracy of the trained model
    """
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='loss', color='orange')
    plt.title('Train Loss and Accuracy')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y', color='orange', labelcolor='orange')
    ax1.set_xlabel('Epoch')

    # secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(history.history['accuracy'], label='accuracy', color='DarkRed')
    ax2.tick_params(axis='y', color='DarkRed', labelcolor='DarkRed')

    plt.show()


def plot_loss_accuracy(history):
    """
    Ploting the history of train and test sets
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()