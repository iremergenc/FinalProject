
def figplot(x,y,arrsize,xlabel,ylabel,title,label,style):
    import matplotlib.pyplot as plt
    if style == 'plt':
        for i in range(arrsize):
            plt.plot(x,y[i],label=label[i])
        plt.grid(which='both',axis='both',linestyle='-.',linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='lower left',fontsize='xx-small')
        plt.title(title)
        plt.show()
    if style =='sct':
        plt.scatter(x, y, c=label)
        plt.grid(which='both', axis='both', linestyle='-.', linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='lower left', fontsize='xx-small')
        plt.title(title)
        plt.show()
    return