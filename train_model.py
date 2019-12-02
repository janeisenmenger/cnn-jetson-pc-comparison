import signal
import os
import sys
import time
import datetime
import getopt
import helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# global vars in case we stop via ctrl+c but still want to save everything
model = None
improvement_dir = None
num_classes=10
start_time = None
test_input = None
test_expected_output = None
timestamp = None

def train(module_name):
    ''' 
    A function to train a model as specified in the module name with the cifar10

    :param module_name: The name of the module we attempt to load in order to get the model.
    :param model_output: The folder in which we want to save the model.
    '''
    # global vars bc we need those later if we want to save upon ctrl+c
    global model, improvement_dir, num_classes, start_time, test_input, test_expected_output, timestamp
    

    # import here so we don't have to load tensorflow before we actually need it
    # 
    import tensorflow.keras as keras
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.callbacks import ModelCheckpoint

    # get training data
    (training_input, training_expected_output), (test_input, test_expected_output) = cifar10.load_data()

    # import the module containing the model.
    model_module = __import__(module_name, fromlist=[''])
    model_container = model_module.CNNModel(training_input.shape[1:])

    # get the models    
    model = model_container.get_model()

    # print automatically
    model.summary()

    # Let's train the model using RMSprop
    model.compile(loss=model_container.get_loss_function(),
              optimizer=model_container.get_optimizer(),
              metrics=['accuracy'])

    # where to save the weight improvements during training
    improvement_dir = 'weights_improvement/' + module_name + '/' + timestamp
    try:
        os.makedirs(improvement_dir)
    except FileExistsError:
        # directory already exists
        pass

    improvement_file_format = improvement_dir + '/{epoch:02d}-{val_loss:.10f}.hdf5'
    
    checkpoint = ModelCheckpoint(
        improvement_file_format, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min')
    callback_list = [checkpoint]

    # split training data 
    
    # Convert class vectors to binary class matrices.
    training_expected_output = keras.utils.to_categorical(training_expected_output, num_classes)
    test_expected_output = keras.utils.to_categorical(test_expected_output, num_classes)

    # normalize
    training_input = training_input.astype('float32')
    test_input = test_input.astype('float32')
    training_input /= 255
    test_input /= 255

    # register the ctrl+c signal handler
    signal.signal(signal.SIGINT, signal_handler)


    start_time = time.time()
    # train
    model.fit(
        training_input, 
        training_expected_output, 
        epochs=model_container.get_epochs(),
        shuffle=True,
        batch_size=model_container.get_batch_size(), 
        validation_data=(test_input, test_expected_output), 
        callbacks=callback_list)
    
    save_all_models()

def print_usage_and_exit():
    '''
    A function to print the usage of the script and then exit.
    '''
    print('Usage: python train_model.py [-m/--module-name=] <module-containing-the-model> [-s/--save-model=] <where-to-save-the-model (dir)>')
    sys.exit(1)

def save_all_models():
    '''
    A function to save all models. The function has no parameters since it uses global variables.
    These have to be global in case this function is being called from the ctrl+c signal handler.
    '''
    global model, improvement_dir, model_output, start_time, test_input, test_expected_output

    end_time = time.time()

    print('Saving all models')

    # The model might have trained further and worsened, therefore we need to load the best weights now.
    weight_files = helper.get_all_files_in_directory(improvement_dir, '.hdf5')
    latest_improvement = max(weight_files, key=os.path.getctime)

    model.load_weights(latest_improvement)
    try:
        os.makedirs(model_output)
    except FileExistsError:
        # directory already exists
        pass
    
    # save best model iteration to file
    model.save(model_output + '/model.hdf5')

    # Score trained model.
    scores = model.evaluate(test_input, test_expected_output, verbose=1)

    # duration 
    with open(model_output + '/parameters.txt', 'w') as file:  
        # trunk file
        file.seek(0)
        file_content = []
        file_content.append("Duration:      " + str(end_time - start_time))
        file_content.append("Test loss:     " + str(scores[0]))
        file_content.append("Test accuracy: " + str(scores[1]))
        #file_content.append("========================================")
        #model.summary(print_fn=lambda x: file_content.append(x))
                
        file.write("\n".join(file_content))

def signal_handler(sig, frame):
    '''
    A signal handler that will save all models and then exit gracefully.

    :param sig: Not used, here for compatability.
    :param frame: Not used, here for compatability.
    '''
    save_all_models()
    sys.exit(0)


if __name__ == "__main__":   
    if len(sys.argv) != 5:
        print_usage_and_exit()

    global model_output

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:s:', ['module-name=', 'save-model-to='])
    except getopt.GetoptError as e:
        # script wasnt called properly, tell the user and exit.
        print(e)
        print_usage_and_exit()
        
    module_name = None

    # extract arguments
    for opt, arg in opts:
        if opt in ('-m', '--module-name'):
            module_name = arg
        elif opt in ('-s', '--save-model-to'):
            model_output = arg
        else:
            print_usage_and_exit()

    if model_output is None:
        print('You have to specify where to save the model to.')
        print_usage_and_exit()                
    elif module_name is None:
        print('Specify the module_name where the model can be found.')
        print_usage_and_exit()

    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    model_output = model_output + '/' + timestamp

    # run the training
    train(module_name, )