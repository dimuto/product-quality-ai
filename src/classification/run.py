from dataloader import Dataloader
from train import Train
from predict import Predict

class Run:
    def run_training(self):
        d = Dataloader()
        dataloaders, dataset_sizes, class_names = d.data_transform()
        t = Train(dataloaders, dataset_sizes, class_names)
        t.run()
    
    def run_prediction(self, test_image_path):
        p = Predict()
        p.predict_model(test_image_path)


if __name__ == "__main__":
    mode = "predict"
    test_image_path = "../data/crops/test/a3.png"

    if mode == "train":
        r = Run()
        r.run_training()
    else:
        r = Run()
        r.run_prediction(test_image_path)