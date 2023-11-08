# Deepfake-Classification

Real vs Fake image classification using InceptionResNet

### Model Selection

|     Model              |     Train_acc (%)    |     Val_acc (%)    |     Test_acc (%)    |
|------------------------|----------------------|--------------------|---------------------|
|     ResNet             |     79.99            |     81.24          |     80.48           |
|     MobileNet          |     80.00            |     80.42          |     80.70           |
|     DenseNet           |     82.23            |     83.04          |     82.58           |
|     InceptionResNet    |     85.89            |     86.69          |     86.48           |

### InceptionResNet Fine tuning

|     Train_acc (%)    |     Val_acc (%)    |     Test_acc (%)    |
|----------------------|--------------------|---------------------|
|     96.74            |     96.68          |     96.71           |

### Feature visualization in InceptionResNet



https://github.com/sanjay-906/Deepfake-Classification/assets/99668976/42b79330-6454-4e66-9318-b0183965df7b


### Output




https://github.com/sanjay-906/Deepfake-Classification/assets/99668976/286431bd-ff8d-4704-873d-23bae36e3e1a


### Future Work
- Video-deepfake instead of frame-by-frame processing
- Thorough image processing
- Object localization instead of face detection + classification
- NAS and Hyperparameter tuning
- Visual Transformers instead of CNNs

### Drawbacks
- Model fails when multiple faces are present in a frame
- High testing accuracy but poor real time performance
- False positive outputs for out-of-context video frames
- High cost for output
- Absence of sequential processing
- 0.5 threshold is not working 
