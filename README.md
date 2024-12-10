# Edge AI Satellite-to-Map Image Translation with Pix2Pix

This repository demonstrates a Pix2Pix-based Generative Adversarial Network (GAN) that translates satellite imagery into corresponding map images. The project aligns with Edge AI and geospatial intelligence workflows, showing how advanced computer vision techniques can be adapted and optimized for deployment on edge devices (e.g., drones, satellites, or other IoT sensors).

## Background & Motivation

**What is Edge AI?**  
Edge AI involves running AI models locally on edge devices with limited computational resources. This approach reduces latency, conserves bandwidth, and improves reliability and data privacy. In geospatial settings, processing data directly at the edge (such as on a satellite or an unmanned aerial vehicle) is critical for real-time decision-making and efficient bandwidth usage.

**Pix2Pix for Geospatial Data:**  
The Pix2Pix architecture uses a conditional GAN to learn a mapping from an input image domain (satellite images) to a target domain (map representations). By training this model, we can convert raw satellite imagery into street map views or other geospatial formats that are easier to interpret or integrate into GIS systems. Potential use cases include:

- Generating map overlays directly on a drone or satellite to reduce reliance on ground infrastructure.
- Quickly producing simplified representations for real-time navigation or situational awareness.
- Serving as a preprocessing step for edge AI pipelines focused on object detection, land-use classification, or environmental monitoring.

## Repository Contents

- **src/**  
  - **data_preparation.py:** Functions to load and preprocess the maps and satellite imagery dataset.  
  - **models.py:** Definitions of the Pix2Pix discriminator, generator (U-Net), and combined GAN model.  
  - **train.py:** Training script for the Pix2Pix model, including loading data, defining optimizers, and executing the training loop.  
  - **inference.py:** Script to load the trained generator model and perform inference on new satellite images.  
  - **utils.py:** Utility functions for plotting results, saving models, and other helpers.
  - **requirements.txt:** Lists Python dependencies.

- **model/**  
  - **model_latest.h5:** A trained generator model file saved after completing training.

- **api/**  
  - **app.py:** A Flask-based API that loads the trained generator model and offers an endpoint to submit a satellite image and retrieve the translated map image. This simulates running a lightweight inference service on edge hardware.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/edge-ai-satellite-to-maps.git
   cd edge-ai-satellite-to-maps
