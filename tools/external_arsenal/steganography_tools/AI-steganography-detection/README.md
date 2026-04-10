# AI-Based Steganography Detector
This repository made for an AI-powered steganography detection system using deep learning. The project includes **training a Convolutional Neural Network (CNN)** on stego images and detecting hidden information inside digital images.  This project can be a great starting component of a Security Operations Center (SOC), Endpoint Detection & Response (EDR), or Security Orchestration, Automation, and Response (SOAR) system, by giving it an automated layer of anti-malware analysis in context of a security framework.


## **Dataset**
We used the **Stego Images Dataset** from **Kaggle**:

🔗 [Stego Images Dataset - Kaggle](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset)

The dataset consists of **clean** and **stego** images, which must be correctly organized into separate folders before training.
If the dataset is not structured properly, use the `organize_data.py` script to arrange the files.


## **Installation**
First, clone the repository:

`git clone https://github.com/7megaumka7/AI-steganography-detection.git`

`cd AI-steganography-detection`


Then, install all required dependencies:

`pip install -r requirements.txt`


## **Training the Model**
To train the model, use the train.py script:

`python train.py`

The trained model will be saved as:

`/PATH/TO/steganography_detector.h5`

The training process will:
- Train a Convolutional Neural Network (CNN)
- Use binary classification (clean vs. stego images)
- Save only the best performing model
- Delete unnecessary files to save storage

## **Using the Detection Script**

Make sure to manually edit the code, especially the part with the condition coefficent, cause it really varies based on the training proccess.

Once the model is trained, run the detect.py script:

`python detect.py`

Input: Enter the image path when prompted.

Output: The script will classify the image as either:

🛑 Steganography Detected

✅ Clean Image


## **Example usage:**

Enter image path: `C:/Users/User/Documents/test_image.png`

Raw Model Output: 0.34.

✅ Clean Image.

## **File Structure:**
/Steganography-Detection

│── train.py 

│── detect.py 

│── organize_data.py 

│── requirements.txt 

│── README.md     

│── /dataset     

## **Requirements:**

Ensure that Python 3.10+ is installed. Use the following command to install dependencies:

`pip install -r requirements.txt`

## **References:**

Kaggle Dataset: [Stego Images Dataset - Kaggle](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset).

TensorFlow Official Documentation: tensorflow.org.

Image Processing with OpenCV: docs.opencv.org.

If you want to find out more about the usage of CNNs, feel free to checkout [our article](https://ieeexplore.ieee.org/document/10811228).

Developed by: Umetaliyev Alisher

📅 Last Updated: February 2025
