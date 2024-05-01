# MULTILINGUAL LANGUAGE DETECTION AND TRANSLATION

# Introduction
The necessity for multilingual communication systems has become increasingly apparent in our globalized society, where interactions across different languages are common. Traditional translation and language detection methods often struggle with accuracy and efficiency, particularly when dealing with a diverse set of languages such as Hindi, Tamil, Gujarati, and Bengali. This project introduces a sophisticated language detection and translation system utilizing state-of-the-art technology in machine learning and artificial intelligence.
This system integrates Convolutional Neural Networks (CNN) with TorchAudio for precise language detection from audio inputs. Following detection, a Transformer model is employed to translate the identified language into English. This approach aims to harness the strengths of advanced neural network architectures for handling complex patterns in speech and text translation, providing a seamless and automated solution for multilingual translation challenges

# Aim and Objectives
To develop a high-performance, scalable, and efficient multilingual language detection and translation system tailored specifically for Hindi, Tamil, Gujarati, and Bengali, translating these into English to facilitate clearer communication.
1. Language Detection
2. Language Translation
3. Testing and Validation
4. Scalability and Deployment


# Value Statement
In today's interconnected world, language barriers can significantly hinder communication and access to information. The development of a robust multilingual language detection and translation system addresses this challenge head-on, offering extensive benefits across several domains:
Educational Access: Students who speak Hindi, Tamil, Gujarati, and Bengali can access educational materials and courses in English, broadening their learning opportunities and resources.
Customer Support: Businesses can provide better support to a diverse customer base, improving user experience and satisfaction by interacting in the customer's native language.
Media Consumption: Enhances the accessibility of media such as films, podcasts, and interviews by providing accurate subtitles and translations in real-time.
Emergency Services: Improves the effectiveness of emergency response in multilingual regions, ensuring that critical information is conveyed accurately regardless of the language spoken by the caller

# Architectures
## 1. CONVOLUTIONAL NEURAL NETWORKS (CNNs)
The CNN in this project is specifically designed to analyze audio data for language detection. The architecture must capture essential linguistic features from complex audio inputs, which include different accents, tones, and speech speeds.

## 2. TORCHAUDIO
TorchAudio plays a crucial role in preparing the audio data before it is fed into the CNN.

## 3. TRANSFORMER MODELS
Transformer models represent a significant advancement in deep learning, particularly in the field of natural language processing (NLP).

## Dataset Description
The "Audio Dataset with 10 Indian Languages" available on Kaggle is a comprehensive collection of audio recordings from ten different Indian languages. This dataset is particularly valuable for projects aimed at developing speech recognition and language detection models because it provides a diverse range of phonetic and linguistic features across multiple Indian languages.
[https://www.kaggle.com/datasets/hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages]

# Methodology
The steps are as follows:
1) Import Required Libraries
2) Using pydub to convert mp3 to wav files for training in      colab
3) Create Metadata for the dataset
4) Custom Dataset class
5) Create a Base model for Image classification (spectrogram image)
6) Build Model Architecture
7) Make stratified split of data to have same number of samples per each class
8) Train the model
9) Train the model for 30 epochs
10) Prediction
11) Converting audio to text

# Evaluation Metrics
## Accuracy
Accuracy is defined as the proportion of true results (both true positives and true negatives) among the total number of cases examined. It is calculated by dividing the number of correct predictions by the total number of predictions made by the model.
Formula:
### Accuracy = Number of correct predictions / Total number of predictions made

## Loss (Cross-Entropy Loss)
Cross-entropy loss, a prevalent loss function for classification problems, measures the performance of a classification model whose output is a probability value between 0 and 1. It quantifies the discrepancy between the predicted probability assigned to the true class and the actual outcome

## F1 Score
The F1 Score is a crucial metric that combines precision and recall of the classifier into a single measure. Precision is the accuracy of positive predictions formulated by the classifier, while recall measures the ability of the classifier to find all the positive samples. The F1 Score is particularly useful when the costs of false positives and false negatives vary, or when there is an imbalance in class distribution.
Formula:
###  F1 Score=2× Precision+Recall / Precision×Recall

# Conclusion
This project set out to develop a robust multilingual language detection and translation system capable of understanding and translating spoken content in Hindi, Gujarati, Tamil, and Bengali into English. Utilizing state-of-the-art technologies such as Convolutional Neural Networks (CNNs) for language detection and Transformer models for translation, the project aimed to bridge communication barriers and enhance accessibility across different linguistic communities.

