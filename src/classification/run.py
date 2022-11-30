import os
from classification.dataloader import Dataloader
from classification.train import Train
from classification.predict import Predict

class Run:
    def run_training(self):
        d = Dataloader()
        dataloaders, dataset_sizes, class_names = d.data_transform()
        t = Train(dataloaders, dataset_sizes, class_names)
        t.run()
    
    def run_prediction(self, test_image_path):
        p = Predict()

        # Image file
        if os.path.isfile(test_image_path):
            defect_prob = p.predict_model(test_image_path)
            pq_score = int(defect_prob * 10)

            return pq_score
        
        # Images in folder
        elif os.path.isdir(test_image_path):
            probs = []
            for i in os.listdir(test_image_path):
                try:
                    defect_prob = p.predict_model(os.path.join(test_image_path, i))
                    probs.append(defect_prob)
                except Exception as e:
                    print(e)
            print(probs)
            avg = sum(probs) / len(probs)
            pq_score = int(avg * 10)

            return pq_score



if __name__ == "__main__":
    mode = "predict"
    test_image_path = "../data/IMG_20190816_154429.jpg"

    if mode == "train":
        r = Run()
        r.run_training()
    else:
        r = Run()
        pq_score = r.run_prediction(test_image_path)
        print(pq_score)