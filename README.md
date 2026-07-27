# Plant Disease Detector

I made this for my village farmers(including my father) who lose crops every year to disease this helps them via their phone through a whatsapp bot which replies in telugu 
NOTE: Here only plant disease detector model is present and everything is related to plant disease detector only

## So What it does

- predicts the plant disease from a photo of a leaf (38 classes)
- gives the treatment
- shows the disease name + treatment in Telugu
- draws a Grad-CAM heatmap so you can see where the model looked

## How it works

- model: EfficientNetV2B0, pretrained on ImageNet (transfer learning)
- trained in 2 phases: frozen base first, then fine-tuned the top layers at a small learning rate (1e-5)
- built in Google Colab with TensorFlow / Keras
- app made with Gradio

## Dataset

- PlantVillage (Kaggle), ~54k images, 38 classes
- split 70/15/15 train/val/test

## Results

- 98% on the test set
- but only ~4 out of 9 correct on my own real-world photos (will do more testing in version 2 becuase there is a flaw in data set not the model)
- So basically in the plant village dataset there are multilple images of the same leafs shot with different angles and duplicated images of the same leaf and i split the dataset at random into 70/15/15 and there is a chance that some images of the same leaf but are differnet images were in the both test and train folders which caused the inflated 98 percent accuracy (i got to know about this when i researched about the huge difference in my accuracy in real world images and test folder)

## Limitations

- it overdiagnoses healthy plants because the healthy training images are clean single lab leaves so a real healthy plant (messy background,whole plant, normal lighting) doesnt match what it learned as an healthyplant and since it has to pick one of the 38 classes it ends up guessing a disease
- it only knows the 38 classes it was trained on.it just picks a random disease closest to the 38 classes even if the real one is not there
- it has no rice diseases  which is the main crop in my region(will add in v2)
- Grad-CAM is low-res (7x7) so it shows the main area, not every spot

## What I might add next

- rice disease classes
- fine-tune on real field photos (PlantDoc)
- a WhatsApp bot so farmers can just send a photo

## How to run

- just open the notebook in Colab (GPU for training)
- it needs a kaggle.json key in Drive to download the dataset
- and run top to bottom; the last cells launch the app

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow / Keras
- **Model:** EfficientNetV2B0 (transfer learning + fine-tuning)
- **Explainability:** Grad-CAM
- **App / UI:** Gradio
- **Image processing:** OpenCV,Pillow
- **Data:** NumPy,Pandas,split-folders
- **Plots / metrics:** Matplotlib,Seaborn,scikit-learn
- **Environment:** Google Colab
- **Dataset:** PlantVillage from (Kaggle)
