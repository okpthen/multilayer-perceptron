import matplotlib.pyplot as plt

	

def make_scatter(loss, acc, val_loss, val_acc):

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	ax1.plot(range(len(loss)), loss, label="Training Loss", color="blue")
	ax1.plot(range(len(val_loss)), val_loss, label="Validation Loss", color="orange")
	ax1.set_title("Loss Over Epochs")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss")
	ax1.legend()
	# ax1.savefig("loss.png")

	ax2.plot(range(len(acc)), acc, label="Training Accuracy", color="green")
	ax2.plot(range(len(val_acc)), val_acc, label="Validation Accuracy", color="red")
	ax2.set_title("Accuracy Over Epochs")
	ax2.set_xlabel("Epochs")
	ax2.set_ylabel("Accuracy")
	ax2.legend()
	plt.savefig("loss_acc.png")