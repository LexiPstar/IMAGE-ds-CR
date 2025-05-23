import matplotlib.pyplot as plt

log_file = "loss_log.txt"

def read_log(file):
    epochs, losses = [], []
    with open(file, 'r') as f:
        for line in f:
            epoch, loss = line.strip().split(',')
            epochs.append(int(epoch))
            losses.append(float(loss))
    return epochs, losses

def plot_log_loss(epochs, losses):
    plt.plot(epochs, losses, marker='o')
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    epochs, losses = read_log(log_file)
    plot_log_loss(epochs, losses)
