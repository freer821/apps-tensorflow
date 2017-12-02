import argparse
from libs.data_handler import DataLoader, DataCreator, DataBatch, DataType
from libs.imagesconverter import createExtractedImages
from tensorflow.python.tools import freeze_graph
from src.emorecognition import EmoRecognition
from src.animalrecognition import AnimalRecognition
from src.test.testmodels import TestEmotionModel, TestAnimalModel

# export
def run_export_emotion():
    freeze_graph.freeze_graph(input_graph="./model/emotion-detect/emotion-model.pbtxt", input_saver="",
                              input_binary=False, input_checkpoint="./model/emotion-detect/emotion-model",
                              output_node_names="emo_prd",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                              output_graph="./model/emotion-detect/emotion-model.pb", clear_devices=True,
                              initializer_nodes="")

def run_export_animal():
    freeze_graph.freeze_graph(input_graph="./model/animal/animal-model.pbtxt", input_saver="",
                              input_binary=False, input_checkpoint="./model/animal/animal-model",
                              output_node_names="animal_prd",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                              output_graph="./model/animal/animal-model.pb", clear_devices=True,
                              initializer_nodes="")

# create images
def run_create_emotion_images():
    dCreator = DataCreator('./data/emotion')
    dCreator.create_images_from_csv('./data/emotion/fer2013.csv', 48)

def run_create_luftbilder():
    createExtractedImages("./data/luftbilder", [255,0,255], "./data/luftbilder/output")

# run tests
def run_test_emotion():
    print('run_test_emotion')
    test_emotion = TestEmotionModel('./model/emotion-detect', './data/emotion/emotion-label.txt')
    test_emotion.initGraph_with_pd()
    test_emotion.predict('./data/emotion/PublicTest/2/img_28714.jpg')

def run_test_animal():
    print('run_test_animal')
    test_animal = TestAnimalModel('./model/animal', './data/animal/animal-label.txt')
    test_animal.initGraph_with_pd()
    test_animal.predict('./data/animal/test/cats/cat.1098.jpg')

# run train
def run_train_emtion():
    dLoader = DataLoader('./data/emotion/emotion-label.txt', 48, 3)
    dLoader.load_images_from_dir('./data/emotion/Training', DataType.Train)
    dLoader.load_images_from_dir('./data/emotion/PrivateTest', DataType.Val)
    emoR = EmoRecognition(dLoader.get_classes(), dLoader.get_img_size(), dLoader.get_img_channels(),
                          './model/emotion-detect')
    dBatchTrain = DataBatch(dLoader.get_tr_images(), dLoader.get_tr_labels(), 64)
    dBatchVal = DataBatch(dLoader.get_val_images(), dLoader.get_val_labels(), 32)
    emoR.train(100, dBatchTrain, dBatchVal)


def run_train_animal():
    dLoader = DataLoader('./data/animal/animal-label.txt', 128, 3)
    dLoader.load_images_from_dir('./data/animal/train', DataType.Train)
    dLoader.load_images_from_dir('./data/animal/test', DataType.Val)
    aniR = AnimalRecognition(dLoader.get_classes(), dLoader.get_img_size(), dLoader.get_img_channels(),
                          './model/animal')
    dBatchTrain = DataBatch(dLoader.get_tr_images(), dLoader.get_tr_labels(), 32)
    dBatchVal = DataBatch(dLoader.get_val_images(), dLoader.get_val_labels(), 32)
    aniR.train(1000, dBatchTrain, dBatchVal)




if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-ex", "--export", required=False,
                    help="export model")
    ap.add_argument("-c", "--create_images", required=False,
                    help="create images")
    ap.add_argument("-tr", "--train", required=False,
                    help="training name to run")
    ap.add_argument("-te", "--test", required=False,
                    help="test model")

    args = vars(ap.parse_args())
    if args['export']:
        export_name = args['export']
        if export_name == 'emotion':
            run_export_emotion()
        elif export_name == 'animal':
            run_export_animal()

    if args['create_images']:
        create_name = args['create_images']
        if create_name == 'luftbilder':
            run_create_luftbilder()
        else:
            run_create_emotion_images()

    if args['train'] is not None:
        train_name = args['train']
        if train_name == 'emotion':
            run_train_emtion()
        elif train_name == 'animal':
            run_train_animal()

    if args['test']:
        test_name = args['test']
        if test_name == 'emotion':
            run_test_emotion()
        elif test_name == 'animal':
            run_test_animal()