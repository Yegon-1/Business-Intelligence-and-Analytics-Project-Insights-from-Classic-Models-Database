import matplotlib.pyplot as plt

def create_visualizations(data, title="Visualization", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Creates a simple bar chart for the given data.
    
    Args:
        data (dict): Dictionary with keys as labels and values as numeric data.
        title (str): Title of the chart.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
    """
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
