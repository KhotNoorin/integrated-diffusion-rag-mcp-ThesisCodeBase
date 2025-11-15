# Integrated Diffusion RAG MCP

**Integrating Diffusion Models with Retrieval-Augmented Generation and Multi-Constraint Prompting for Multimodal Content Generation**

---

## Overview

This project integrates **Diffusion Models**, **Retrieval-Augmented Generation (RAG)**, and **Multi-Constraint Prompting (MCP)** to build a unified multimodal content generation system.  
It combines *image generation, text retrieval, and adaptive constraint control* to generate coherent and high-quality visual and textual outputs.

---

## Key Features

- **Diffusion-based Image Generation** – Uses Stable Diffusion pipelines from the Hugging Face `diffusers` library.
- **Retrieval-Augmented Generation (RAG)** – Retrieves contextually relevant information using FAISS or Chroma vector databases.
- **Multi-Constraint Prompting (MCP)** – Dynamically adapts generation parameters based on user-defined constraints.
- **Text + Image Embeddings** – Powered by `transformers` and `sentence-transformers`.
- **Evaluation Metrics** – Includes FID, CLIPScore, and other visual/textual metrics.
- **Interactive UI** – Streamlit-based frontend for easy experimentation and visualization.

---

## Installation

### 1️. Clone the repository

git clone https://github.com/KhotNoorin/integrated_diffusion_rag_mcp.git
cd integrated_diffusion_rag_mcp

### 2️. Create a virtual environment (recommended)

python -m venv .venv
.\.venv\Scripts\activate

### 3️. Install dependencies
If you want the latest compatible versions for Python 3.13+:

pip install -r requirements.txt
Or install the project in editable (development) mode:

pip install -e .

### Usage
Run the frontend app

streamlit run frontend/app.py
Or run as a Python module

python -m frontend.app

Once launched, you can:
- Enter a text prompt
- Adjust generation constraints (e.g., max tokens, creativity)
- Generate multimodal outputs (images + text)
- View evaluation results interactively

## Technologies Used
- Python 3.13
- PyTorch, Diffusers, Transformers
- FAISS, ChromaDB
- Streamlit / Gradio
- Pandas, NumPy, Matplotlib
- Torchmetrics, Scikit-image

## Developer Notes
- Works on Windows, macOS, and Linux
- Fully supports CPU and CUDA GPU acceleration
- Modular design – easily extendable with new RAG retrievers or diffusion backbones
- For Intel optimization, you can later add intel-extension-for-pytorch once Python 3.13 support is available

## Example Workflow
1. Provide a prompt (e.g., “A futuristic city skyline at sunset”)
2. Adjust MCP parameters in the sidebar:
    - Max token length
    - Creativity/temperature
    - Constraint weighting
3. The model:
    - Uses RAG to retrieve contextually related text or image embeddings
    - Applies MCP to enforce generation constraints
    - Feeds the processed prompt to the Diffusion Model for final generation
4. Outputs:
    - Generated image(s)
    - Textual summaries or captions
    - Evaluation metrics such as FID and CLIPScore

##Evaluation

You can evaluate generated samples using:

python -m evaluation.run_metrics

### Supported metrics:
- FID (Fréchet Inception Distance)
- CLIPScore
- Cosine Similarity
- Human Evaluation Reports (Excel via openpyxl)

---

## Acknowledgements

Built using:
- Hugging Face Diffusers
- Transformers
- FAISS
- Streamlit

----

## Author

Noorin Khot
GitHub: KhotNoorin

---
