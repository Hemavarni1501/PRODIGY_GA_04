# GA_04: Image-to-Image Translation with cGAN (Pix2Pix)

## 🔗 Project Links
* **Live Application:** [Pix2Pix Translation Engine](https://appigyga04-hauwortdzgqv6urknigxup.streamlit.app/)
---

## 📌 Project Overview
This project involves the implementation of a **Conditional Generative Adversarial Network (cGAN)**, specifically the **Pix2Pix** architecture, as part of the **Prodigy Infotech** Generative AI internship. 

While traditional GANs generate images from random noise, **Pix2Pix** is an image-conditional model. It learns a mapping from an input domain (e.g., sketches, label maps, or black-and-white photos) to a target output domain (e.g., photo-realistic images or colorized photos).

---

## 🛠️ Technical Architecture

### 1. The Generator (U-Net)
The generator is responsible for translating the input image. It utilizes a **U-Net** architecture, which is an encoder-decoder network with **Skip Connections**. 
* **Encoder:** Downsamples the image to extract high-level feature representations.
* **Decoder:** Upsamples the features back to the original resolution.
* **Skip Connections:** These bridge the encoder and decoder layers, allowing the network to shuttle low-level spatial information across the bottleneck, preventing the loss of fine structural details.

### 2. The Discriminator (PatchGAN)
Unlike standard GAN discriminators that output a single "real/fake" value for the entire image, Pix2Pix uses a **PatchGAN**.
* It classifies individual $N \times N$ patches of an image as real or fake.
* This focuses the model on **high-frequency details** (textures and crisp edges), leading to sharper, more realistic results.

### 3. Loss Functions
The model is optimized using a weighted combination of:
* **Adversarial Loss:** Encourages the generator to produce images that are statistically indistinguishable from the target domain.
* **L1 Loss (Mean Absolute Error):** Measures the pixel-wise distance between the generated image and the ground truth, ensuring structural alignment and reducing blurring.

---

## 🚀 Key Features
1. **Interactive UI:** Built with **Streamlit** to allow users to upload input images and witness real-time image translation.
2. **Normalized Pipeline:** Implements the standard GAN preprocessing pipeline, normalizing pixel values to the $[-1, 1]$ range for training stability.
3. **Downloadable Results:** Users can instantly download the synthesized output for further use.

---

## 📁 Installation & Local Setup
To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Hemavarni1501/PRODIGY_GA_04](https://github.com/Hemavarni1501/PRODIGY_GA_04).
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
### Internship: Prodigy Infotech | Track: Generative AI | Task: 04

#### Developer: Hemavarni.S
