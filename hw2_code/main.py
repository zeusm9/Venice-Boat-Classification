import clean_data as cd
import train
import classify as cl
def main():
    print("Begin...")
    filename = "ground_truth.txt"
    cd.clean_truth(filename)
    cd.clean_test()
    train.train("vgg")
    accuracy,y_true,y_pred = cl.test("test","new_ground_truth.txt")
    accuracy,precision,recall,f1 = cl.get_metrics(y_true,y_pred)
    print("Accuracy: %s\nPrecision: %s\nRecall: %s\nF1-Score: %s\n" % (accuracy,precision,recall,f1))

if __name__ == "__main__":
    main()
