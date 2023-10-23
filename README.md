# Swahili_Classification_Project
# SWAHILI AUDIO PREDICTION: AN AUTOMATIC SPEECH RECOGNITION (ASR) PREDICTIVE MODELING PROJECT
## Introduction
![image](Readme_image.jpeg)

Swahili, originating on the East African coast, acted as a significant lingua franca, influenced by interactions between Bantu communities and Arab traders. Thriving city-states like Kilwa and Zanzibar boosted its prominence. In Automatic Speech Recognition (ASR) projects, Swahili's unique phonetics and dialects are a compelling challenge. ASR aims to transcribe spoken Swahili into text, making it useful for transcription and translation. Successful ASR for Swahili requires tailored models that adapt to the language's nuances. This technology plays a vital role in preserving and promoting Swahili's linguistic and cultural heritage, fostering cross-cultural communication and expanding the language's utility.

## Problem Statement
Swahili, also known as Kiswahili, has a rich history that spans centuries and is now one of the most widely spoken languages in Africa, with millions of speakers across various countries. Today, Swahili is not only a language of communication but also a symbol of cultural heritage and identity for millions of people in East Africa and beyond. Its history reflects the dynamic nature of language, shaped by trade, migration, colonization, and cultural exchange over the centuries.
With the increasing availability of digital audio content in Swahili, AnalytiX Insights aims to develop automated systems that can classify and categorize Swahili audio recordings for various applications, including speech recognition, content recommendation, and language learning tools.

## Main Objective
To develop an automated system for converting basic Swahili audio into written text using speech recognition technology.

## Specific Objectives
1. To develop a machine learning model capable of translating Swahili audio recordings.
2. To deploy a model that transcribes the recorded audio files.
3. To provide recommendations for further enhancements and applications.


## Experimental Designing

* Data Collection
* Data Preprocessing
* Exploration Data Analysis
* Feature Extraction
* Modelling
* Evaluation
* Deployment
* Conclusion
* Recommendations

## Metric of Success
Accuracy:
* The ratio of correctly recognized words to the total number of words in the reference transcription. A high accuracy rate will indicate a successful ASR system, as it signifies the system's capability to understand and convert spoken language into text with a high level of correctness.

Word Error Rate[WER]
* It quantifies the accuracy of the system by comparing the transcribed text to the reference text and measuring the number of insertions, deletions, and substitutions required to align them. A lower WER will indicate a higher level of success, as it represents a closer match between the system's output and the expected transcript.


Data Understanding
The data used in this project was collected by 300 contributors based in Kenya. It consists of recordings of twelve different phrases spoken in Swahili.Here are the 12 words and their English translations. You need to predict the Swahili word; the English is here for interest's sake.
| SWAHILI  | ENGLISH  |
| -------- |:--------:|
| Ndio     | Yes      |
| Hapana   | No       |
| Mbili    | Two      |
| Tatu     | Three      |
| Nne     | Four      |
| Tano     | Five      |
| Sita     | Six      |
| Saba     | Seven      |
| Nane     | Eight      |
| Tisa     | Nine      |
| Kumi     | Ten      |
| Moja     | One      |



## Conclusions

**AlexNet ASR Model**

Training Loss: The training loss decreased over the epochs, indicating that the model was learning and converging. However, the validation loss seems to be consistently higher than the training loss, which could suggest that the model might be overfitting to the training data.

Accuracy: The accuracy for the AlexNet ASR model is very low, around 0.07. This suggests that the model is not performing well in terms of correctly transcribing audio recordings. There might be issues with the model architecture, data preprocessing, or training process.

Learning Rate Scheduling: The learning rate was adjusted during model training. It helps the model converge efficiently.

**ResNet18 ASR Model**

Training Loss: The training loss for the ResNet18 ASR model decreased over the epochs, indicating that the model learned the data well. Like the AlexNet model, the validation loss is consistently higher, indicating a potential overfitting issue.

Accuracy: The accuracy for the ResNet18 ASR model is much higher, around 0.937, which is a significant improvement compared to the AlexNet model. This suggests that the ResNet18 architecture might be better suited for the ASR task.

Word Error Rate (WER): A WER of 0.063 indicates that the ResNet18 model is making relatively few errors in transcribing the audio recordings. This is a positive sign of its performance.


## Recommendations

* Further Model Evaluation: While ResNet18 shows promising results, it's essential to perform more rigorous evaluation, including testing on a larger and diverse dataset. This can help assess the model's generalization ability.

* Incorporate a broader range of Swahili audio recordings, including longer sentences and passages to expand the scope of this project.

* Language Model Integration: To enhance transcription accuracy, consider incorporating language models to improve the fluency and coherence of the transcribed text.

* Data Quality and Preprocessing: Ensure that the data used for training and validation is of high quality and that proper preprocessing techniques are applied. Data augmentation methods can also be employed to enhance the model's ability to handle various audio conditions and audio formarts.

* Use a powerful GPU device to train the model on big audio files and improve the models performance.



