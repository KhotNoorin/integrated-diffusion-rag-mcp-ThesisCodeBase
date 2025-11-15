from setuptools import setup, find_packages

setup(
    name="integrated_diffusion_rag_mcp",
    version="1.0.0",
    author="Noorin Khot",
    description="Integrating Diffusion Models with Retrieval-Augmented Generation and Multi-Constraint Prompting for Multimodal Content Generation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KhotNoorin/integrated_diffusion_rag_mcp",
    license="MIT",
    packages=find_packages(exclude=("tests", "notebooks", "experiments")),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        # ===== Basics & Utilities =====
        "python-dotenv",
        "PyYAML",
        "tqdm",
        "loguru",
        "rich",

        # ===== Core ML & Embeddings =====
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "sentence-transformers",

        # ===== Retrieval & Indexing =====
        "faiss-cpu",
        "chromadb",
        "scikit-learn",

        # ===== Diffusion & Image Generation =====
        "diffusers[torch]",
        "accelerate",
        "safetensors",

        # ===== Evaluation Metrics & Visualization =====
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-image",
        "torchmetrics",

        # ===== UI / Frontend =====
        "streamlit",
        "gradio",
        "Pillow",
        "markdown-it-py",

        # ===== Human Eval & Reporting =====
        "openpyxl",
        "latexcodec",

        # ===== Optional: ControlNet / Fusion Modules =====
        # "peft",  # install manually via: pip install git+https://github.com/huggingface/peft.git
        "onnxruntime",

        # ===== Misc & Dev =====
        "requests",
        "pytest-mock",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
            "mypy",
            "pre-commit",
            "Sphinx",
            "mkdocs",
            "mkdocs-material",
        ]
    },
    entry_points={
        "console_scripts": [
            "integrated_demo=frontend.app:main",
            "run_experiments=experiments.run_experiment:main",
            "run_ablation=experiments.run_ablation:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
