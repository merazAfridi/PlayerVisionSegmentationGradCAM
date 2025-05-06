## PlayerVisionSegmentationGradCAM

This project uses computer vision to find and highlight football players in pictures. It uses a method called Grad-CAM to show how the computer makes its decisions. The goal is to help understand player actions and performance better by making the computer's decisions easy to see.
#### Homepage
![Homepage](https://github.com/merazAfridi/PlayerVision-Football-Players-Segmentation-Website-/blob/main/homepage%20demo.PNG)
#### Result
![Results](https://github.com/merazAfridi/PlayerVision-Football-Players-Segmentation-Website-/blob/main/Result%20Page%20demo.PNG)
#### About Me
![About Me](https://github.com/merazAfridi/PlayerVision-Football-Players-Segmentation-Website-/blob/main/about%20me%20page%20demo.PNG)

### Dataset Description

This project utilizes two datasets related to football player segmentation for computer vision tasks:

---

[Football Player Segmentation Dataset](https://www.kaggle.com/datasets/ihelon/football-player-segmentation)

   This dataset contains images of football players in various playing positions, such as goalkeepers, defenders, midfielders, and forwards. The images are captured from different angles and distances. Each image is annotated with pixel-level masks that specify player locations and segmentation boundaries.



   **Use Cases:**  
   - Player detection and segmentation during matches.  
   - Analyzing player movements and behaviors.  
   - Exploring performance metrics and trends based on positional data.
     
![Dataset Image](https://github.com/merazAfridi/PlayerVision-Football-Players-Segmentation-Website-/blob/main/static/results/f1ed0910592644f3b0cb340f41ee6d9c_resized.png)

---

This mask dataset, based on the  Football Players Segmentation dataset by Yaroslav Isaienkov, contains 512 x 512 images with annotations converted from COCO JSON to JPG format

---
#### Image, Predicted MASK & GradCAM Analysis
![Dataset Image Prediction](https://github.com/merazAfridi/PlayerVision-Football-Players-Segmentation-Website-/blob/main/evaluation.png)

---
#### Matrics Evaluation
![matrics Evaluation](https://github.com/merazAfridi/PlayerVision-Football-Players-Segmentation-Website-/blob/main/Model%20Evaluation%20(Unet_ResNet50).png)

---

### Attribution
Special thanks to Yaroslav Isaienkov for the Football Players dataset and Rafi Darmawan for the corresponding mask annotations.
Please refer to the respective links for detailed licensing and usage terms provided by the authors.

For access to the full dataset and saved models, please reach out to **Gazi Meraz Mehedi Afridi** at [meraz.afridi@gmail.com](mailto:meraz.afridi@gmail.com)

