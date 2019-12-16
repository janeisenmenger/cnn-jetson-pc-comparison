import sys
import getopt
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def measure_fps(model_file):
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.models import load_model

    model = load_model(model_file)

    # ignore training date, only take validation input
    # we don't care about accuracy, since we already measured that during training
    _, (input_data, _) = cifar10.load_data()

    print("Start measuring fps.")
    start_time =time.time()

    model.predict(input_data, batch_size=1)
    end_time = time.time()

    time_spent = end_time - start_time

    print("Finished. Average fps: " + str(len(input_data)/time_spent))


def print_usage_and_exit():
    '''
    A function to print the usage of the script and then exit.
    '''
    print('Usage: python measure.py [-m/--module-file=] <model_file.hdf5>')
    sys.exit(1)

if __name__ == "__main__":   

    if len(sys.argv) != 3:
        print_usage_and_exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:', ['model-file='])
    except getopt.GetoptError as e:
        # script wasnt called properly, tell the user and exit.
        print(e)
        print_usage_and_exit()
        
    model_file = None

    # extract arguments
    for opt, arg in opts:
        if opt in ('-m', '--model-file'):
            model_file = arg
        else:
            print_usage_and_exit()

    if model_file is None:
        print('You have to specify which model you want to measure.')
        print_usage_and_exit()                

    # run the training
    measure_fps(model_file, )