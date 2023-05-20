import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def learning_curve(model, X, y):
    """
    Return the mean values of the train and test scores
    """
    train_sizes = np.arrange(0, len(X) * 0.8)
    # Get train scores (R2), train sizes, and validation scores using `learning_curve`
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model, X=X, y=y, train_sizes=train_sizes, cv=5)

    # Take the mean of cross-validated train scores and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    return train_scores_mean, test_scores_mean


def learning_curve_plot(model, X, y):
    """
    Returns the learning curves figure
    Uses the result of learning curve figure
    """
    train_scores_mean, test_scores_mean = learning_curve(model, X, y)

    plt.plot(train_sizes, train_scores_mean,
             label = 'Training score', color='midnightblue')
    plt.plot(train_sizes, test_scores_mean,
             label = 'Test score', color='firebrick')
    plt.ylabel('$R^{2}$ score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves', fontsize = 18, y = 1.03)
    plt.legend()

def vizu_num_features(X_num):
    """
    Receives a dataframe with only numerical features
    and do plots of histogram, boxplot and qqplot

    need to be tested
    """
    # Creating three subplots per numerical_feature
    fig, ax =plt.subplots(len(X_num.columns),3,figsize=(15,3))

    for i, numerical_feature in enumerate(X_num.columns):
        # Histogram to get an overview of the distribution of each numerical_feature
        ax[i, 0].set_title(f"Distribution of: {numerical_feature}");
        sns.histplot(data = X_num, x = numerical_feature, kde=True, ax = ax[0]);

        # Boxplot to detect outliers
        ax[i, 1].set_title(f"Boxplot of: {numerical_feature}");
        sns.boxplot(data = X_num, x = numerical_feature, ax=ax[1]);

        # Analyzing whether a feature is normally distributed or not
        ax[i, 2].set_title(f"Gaussianity of: {numerical_feature}");
        qqplot(X_num[numerical_feature],line='s',ax=ax[2]);

        fig.savefig('num_features_dist.pdf', dpi=300)
