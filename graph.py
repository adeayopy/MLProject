import matplotlib.pyplot as plt


# This method is used to plot the loss and accuracy graph
def plotgraph(location):
    episode=[]
    losses=[]
    accuracy=[]
    with open(location+'/console.log', 'r') as file:
        # Search for line that has the word Eval
        target_word = 'Eval'
        for line in file:
            if target_word in line:
                z=line.strip().split(' ')
                a=line.strip().split(' ')[4]
                # Extract and append loss, episode and accuracy to ther respective list
                try:
                    epi=int(a[:-1])
                    loss=float(z[10])
                    acc=float(z[14])

                    episode.append(epi)
                    losses.append(loss)
                    accuracy.append(acc)                    
                except:
                    a=line.strip().split(' ')[3]
                    epi=int(a[1:-1])
                    loss=float(z[9])
                    acc=float(z[13])

                    episode.append(epi)
                    losses.append(loss)
                    accuracy.append(acc)



    # Create a figure with two y-axes
    fig, ax1 = plt.subplots()

    # Plot the accuracy line graph on the first y-axis
    ax1.plot(episode, accuracy, color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Accuracy', color='blue')

    # Create a loss y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the loss line graph on the second y-axis
    ax2.plot(episode, losses, color='red')
    ax2.set_ylabel('Losses', color='red')
    plt.title('Validation accuracy and loss')
    # Show the plot
    plt.savefig(location+'.png')
