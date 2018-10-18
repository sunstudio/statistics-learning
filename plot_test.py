import matplotlib.pyplot as plt

decision_node = {'boxstyle':'sawtooth'}
leaf_node={'boxstyle':'round4','fc':'0.8'}
arrow = {'arrowstyle':'->'}

axis1 = None


def plot_node(text,center,parent,node_type):
    global axis1
    axis1.annotate(text, xy=parent, xytext=center, xycoords='axes fraction', bbox=node_type, arrowprops = arrow)


def create_plot():
    global axis1
    fig = plt.figure(1)
    fig.clf()
    axis1 = plt.subplot(111)
    axis1.text(0.5, 0.5, 'some text', verticalalignment="top", horizontalalignment="center")
    plot_node('a decision node',(0.5,0.1),(0.1,0.5),decision_node)
    plot_node('a leaf node',(0.8,0.1),(0.3,0.8),leaf_node)
    axis1.annotate('hello', xy=(0.5,0.6), xytext=(0.6, 0.7), bbox={}, arrowprops={ 'arrowstyle':'<->' })
    plt.show()


create_plot()

