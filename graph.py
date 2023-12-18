import matplotlib.pyplot as plt


def plot_outliers(df, selected_columns, outlier_df):
    for feature in selected_columns:
        plt.scatter(df[feature], df.index, label='Normal')
        plt.scatter(outlier_df[feature], outlier_df.index, color='r', label='Outlier')
        plt.xlabel(feature)
        plt.ylabel('Data Point')
        plt.legend()
        plt.show()
