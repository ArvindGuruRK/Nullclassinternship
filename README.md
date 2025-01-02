# Emotion_detection

### **Emotion Detection Using Machine Learning**

#### **Introduction:**
Emotion detection plays a crucial role in various real-world applications, from enhancing user experiences to improving human-computer interactions. This project focuses on developing a machine learning-based solution to identify human emotions through text or other data modalities such as audio or visual inputs.

#### **Project Overview:**
The primary goal of this project was to create a system that can classify emotions using machine learning algorithms. Key steps in the project included data collection, preprocessing, feature extraction, and model training. Various machine learning algorithms were implemented and evaluated to identify emotions such as happiness, sadness, anger, surprise, and more.

#### **What I Created:**
Through this project, I built an emotion detection model capable of accurately analyzing input data and classifying emotions. The project involved:
- **Data Processing:** Cleaning and organizing the dataset to ensure optimal model performance.
- **Feature Engineering:** Extracting relevant features that effectively represent emotional patterns.
- **Model Evaluation:** Using performance metrics like accuracy, precision, and recall to evaluate the models and select the best-performing algorithm.

#### **Outcomes:**
- Successfully trained a machine learning model capable of detecting emotions with high accuracy.
- Demonstrated the effectiveness of different algorithms in emotion recognition tasks.
- Gained hands-on experience in data preprocessing, feature extraction, and the practical application of machine learning models.
- Enhanced my understanding of emotion recognition and its potential real-world applications in fields such as customer service, healthcare, and sentiment analysis.

# TASK 1
# Visualize Activation Maps

For my internship tasks related to the Emotion Detection project, Task 1 focused on visualizing activation maps to gain insights into the image regions that activate specific filters in the pre-trained model for emotion detection. This task aimed to enhance interpretability by identifying the features and patterns the model relies on for classifying emotions. Using the pre-trained model, I generated activation maps to analyze the areas of interest, such as facial landmarks, expressions, and subtle texture variations that contribute to emotion recognition. Key features extracted included the position and shape of the eyes, mouth, and eyebrows, as well as overall facial structure and symmetry. This task provided a deeper understanding of the model's decision-making process and helped to validate the effectiveness of the extracted features in detecting emotions. A GUI was not required for this task, allowing a streamlined focus on model visualization and analysis.


![Screenshot 2024-12-27 112557](https://github.com/user-attachments/assets/54d3b4b7-3452-4496-8707-4ab7ef3e737c)



#### **CNN ACTIVATION MAPS THAT EXTRACTS THE HIGHLY FEATURED IMAGE:**


![Screenshot 2024-12-27 112229](https://github.com/user-attachments/assets/bd41a719-edea-44e3-bfcb-823948e1b3b3)



# TASK 2
# Attendance System Model

For my internship tasks related to the **Emotion Detection** project, Task 1 focused on visualizing activation maps to gain insights into the image regions that activate specific filters in the pre-trained model for emotion detection. This task aimed to enhance interpretability by identifying the features and patterns the model relies on for classifying emotions. Using the pre-trained model, I generated activation maps to analyze the areas of interest, such as facial landmarks, expressions, and subtle texture variations that contribute to emotion recognition. Key features extracted included the position and shape of the eyes, mouth, and eyebrows, as well as overall facial structure and symmetry. This task provided a deeper understanding of the model's decision-making process and helped to validate the effectiveness of the extracted features in detecting emotions. A GUI was not required for this task, allowing a streamlined focus on model visualization and analysis. 


![Screenshot 2024-12-27 100054](https://github.com/user-attachments/assets/e7eb30e3-31d9-45b3-b110-0f2da7327e44)

![Screenshot 2024-12-27 094550](https://github.com/user-attachments/assets/fa205f6a-28f4-4b7f-a522-767dae1a0d51)

![Screenshot 2024-12-27 094527](https://github.com/user-attachments/assets/c11e797d-d5b9-4dc9-9675-6cb3fac038f9)

![Screenshot 2024-12-27 094450](https://github.com/user-attachments/assets/0c2449d3-e4d8-49ea-960d-cae939da22f4)

![Screenshot 2024-12-27 094052](https://github.com/user-attachments/assets/e313910b-bf52-4677-abeb-dd4a57bca691)

![Screenshot 2024-12-27 093904](https://github.com/user-attachments/assets/9e2a6d8b-d532-4304-9d7f-a3ff256a2839)

#TASK 3
# Emotion Detection Through Voice

![Screenshot 2025-01-02 092907](https://github.com/user-attachments/assets/97bf92c5-2f0a-4a23-af0b-99cd09a308da)


This project focuses on detecting human emotions from voice data using machine learning. The model processes audio files to classify emotions such as **angry**, **sad**, **happy**, **fear**, and **surprise**. It supports both pre-recorded voice notes and real-time voice uploads, specifically designed for female voices.


## Introduction
The goal of this project is to build a machine learning model capable of detecting emotions through audio signals. By leveraging advanced audio processing and diverse datasets, the system achieves reliable emotion recognition.

---

## Libraries Used
The following libraries were used to build and implement the project:

- **Librosa**: Audio signal analysis and feature extraction.
- **Audio Libraries**: For handling audio files.
- **Keras**: For building and training deep learning models.
- **NumPy, Pandas**: For numerical operations and data manipulation.
- **Matplotlib, Seaborn**: For data visualization.

![Screenshot 2025-01-02 131743](https://github.com/user-attachments/assets/e63dd46e-c1ae-4d36-828b-4a59ac352d70)


---

## Datasets
The project utilized the following datasets for training and evaluation:
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
- **TESS (Toronto Emotional Speech Set)**
- **SAVEE (Surrey Audio-Visual Expressed Emotion)**

These datasets were combined to ensure diversity and robustness.

---

## Data Exploration and Visualization
- **Emotion Distribution**: Visualized the count of emotions in the dataset to ensure balance.
- **Waveplots**: Generated waveforms for each emotion to study audio signal patterns.
- **Spectrograms**: Visualized frequency and intensity variations over time for deeper insights.

![Screenshot 2025-01-02 130743](https://github.com/user-attachments/assets/e182391d-0c8b-4a2d-9f96-86b48cedc39c)


---

## Data Preprocessing and Augmentation
### Steps:
1. **Data Cleaning**: Standardized audio files to a uniform format and duration.
2. **Data Augmentation**: Applied techniques like:
   - Pitch Shifting
   - Time Stretching
   - Adding Background Noise
3. **Feature Extraction**: Extracted features such as:
   - Mel-frequency cepstral coefficients (MFCCs)
   - Chroma features
   - Spectral contrast

---

## Model Building and Training
The deep learning model was built using the **Keras Sequential API** with the following architecture:
- **Convolutional Neural Networks (CNN)** layers for feature extraction.
- **Dense Layers** for classification.

### Training:
- **Hyperparameters**:
  - Learning Rate: Optimized for best performance.
  - Batch Size: Chosen based on dataset size.
  - Epochs: Determined through experimentation.

 ![Screenshot 2025-01-02 130638](https://github.com/user-attachments/assets/a7b40a87-e0b4-405b-a0ff-0d54deae7a45)

---

## Model Performance
- **Accuracy**: The model achieved an overall accuracy of **75%**.
- **Loss and Accuracy Plots**: Training and validation loss were visualized to monitor model performance over epochs.

![Screenshot 2025-01-02 130619](https://github.com/user-attachments/assets/634895b3-aec1-4cbb-a021-8d4fb9015a22)


---

## Conclusion
This project successfully implemented a system for emotion detection through voice. It demonstrated the potential of audio-based emotion recognition using deep learning, achieving robust results through effective data processing and augmentation. Future improvements could include:
- Extending support for male voices.
- Enhancing model accuracy with additional datasets and advanced techniques.

![Screenshot 2025-01-02 092155](https://github.com/user-attachments/assets/32500ccb-f4de-4c93-9013-ec04abf09ad4)

![Screenshot 2025-01-02 092139](https://github.com/user-attachments/assets/79dbd786-ea5c-46c2-86a9-496f00825d4a)
