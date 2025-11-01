import matplotlib.pyplot as plt

def custom_plot(title, xlabel, ylabel, figsize=(5, 3)):
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)