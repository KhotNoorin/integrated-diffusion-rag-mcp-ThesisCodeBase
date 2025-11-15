# 🧩 Raw Data

This folder stores **unprocessed datasets** before any tokenization or embedding.

Typical sources:
- **COCO Captions Dataset**
- **LAION-400M / LAION-Aesthetics**
- **OpenImages** (for multimodal examples)

To reproduce results:
1. Download the required dataset and place under `data/raw/`
2. Update `config.yaml` → `paths.raw_data`
3. Run `training/dataset_loader.py` to preprocess.

Example command:
```bash
python training/dataset_loader.py --input data/raw --output data/processed

---

# `data/README_data.md`
Documentation for your data pipeline.

```markdown
# Data Directory Structure

This directory contains all datasets and derived resources for the thesis project:
**Integrating Diffusion Models with RAG and Multi-Constraint Prompting for Multimodal Content Generation**

---

## Folders Overview

| Folder | Description |
|---------|--------------|
| `raw/` | Original, unprocessed datasets (images, captions, metadata) |
| `processed/` | Cleaned, tokenized, resized, or standardized data |
| `embeddings/` | Precomputed CLIP/BERT embeddings for retrieval |
| `metadata/` | Dataset info, captions index, mappings |
| `retrieval_index/` | FAISS/Chroma persistent retrieval stores |
| `examples/` | Small demo prompts and images used in Streamlit UI |

---

## Typical Workflow

1. **Collect raw data** → store under `data/raw/`
2. **Run preprocessing**
   ```bash
   python training/dataset_loader.py
→ Outputs to data/processed/
3. **Generate embeddings**
    ```bash
    python models/retrieval/embedder.py
→ Saves .npy and metadata to data/embeddings/
4. **Build FAISS/Chroma index**
    ```bash
    python models/retrieval/index_builder.py
→ Outputs to data/retrieval_index/
5. **Use demo prompts**
    ```bash
    streamlit run frontend/app.py
→ Uses data/examples/demo_prompts.json for fast testing

## Supported Datasets

- You can easily extend the pipeline with:
- COCO Captions (2017)
- LAION-Aesthetics v2
- Conceptual Captions 3M
- WikiArt / ArtBench
- Custom academic or medical datasets
- Update config.yaml to change the data paths and embedding settings.

## Notes

- Keep large datasets (like LAION) out of version control.
- Only small sample subsets (≈ 10–50 images) should remain for reproducibility.
- Ensure all metadata files are UTF-8 encoded.


---

✅ **Usage Summary**
| Component | Purpose | Used by |
|------------|----------|---------|
| `raw/` | Original datasets | `dataset_loader.py` |
| `processed/` | Cleaned datasets | training, evaluation |
| `embeddings/` | CLIP/BERT vectors | retriever, RAG |
| `metadata/` | Index + info | evaluator, constraints |
| `retrieval_index/` | FAISS/Chroma | retriever |
| `examples/` | Demo prompts/images | frontend, demo notebook |

---

Would you like me to now generate **sample preprocessing scripts** —  
`data_preprocessor.py` (for text/image cleaning, tokenization, resizing, metadata linking)  
and  
`embedding_generator.py` (for building text+image embeddings using CLIP/BERT)?
