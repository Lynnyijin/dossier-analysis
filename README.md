# Data Segmentation — Document Classification for Digitised Dossiers

Machine-learning pipeline that segments and classifies pages within digitised immigration dossiers (NAMA collection). Three classification tasks are addressed: detecting document start pages, predicting layout type, and predicting functional category. Each task is explored with up to 12 models spanning three modalities (visual, text, multimodal).

---

## Repository Structure

```
.
├── dossier_composition_analysis.py      # Descriptive statistics & plots for the corpus
├── page_classifier_features.py          # Shared feature extractors (VGG-16, BERT)
├── pdf_to_png.py                        # Render PDF pages to PNG (PyMuPDF)
├── xml_extraction.py                    # Extract page text from PageXML files
│
├── page_start_classifier.ipynb          # Task 1 – binary start-page detection
├── function_start_page_classifier.ipynb # Task 2a – functional category (start pages + propagation)
├── function_full_doc_classifier.ipynb   # Task 2b – functional category (all pages)
├── layout_start_page_classifier.ipynb   # Task 3a – layout type (start pages only)
├── layout_start_page_classifier.ipynb   # Task 3b – layout type (all pages only)
├── outputs/                             # Generated CSVs, plots, and summary report
├── feature_cache/                       # Cached VGG-16 / BERT / EfficientNet-B0 features (.npz)
└── requirements.txt
```

---

## Python Scripts

### `dossier_composition_analysis.py`
Produces descriptive statistics at two levels of granularity and writes all results to `outputs/`.

- **Dossier level** (all 849 dossiers): page-count mean, median, SD, min/max, quartiles, skewness, and a histogram.
- **Document level** (65 annotated dossiers): distribution of functional categories, layout types, and pages-per-document segment. Aggregates four annotation files with majority-vote consensus and emits CSVs, bar charts, and a plain-text summary report.

### `page_classifier_features.py`
Reusable feature-extraction classes imported by all classifier notebooks.

- `VGG16FeatureExtractor` — 4096-D ImageNet VGG-16 penultimate-FC features from page PNGs (batched, GPU-aware).
- `BERTTextFeatureExtractor` — 768-D BERT `[CLS]` embeddings from per-page `.txt` files; maps image paths to their corresponding text files automatically.

### `pdf_to_png.py`
Renders each page of every PDF in the workspace to a PNG file at 150 DPI (configurable) using PyMuPDF. Output is written to `pdf_pages_png/<pdf_stem>/<pdf_stem>_page_XXXX.png`.

### `xml_extraction.py`
Extracts plain text from PageXML files in `NAMA_digitised_page_files/` using the `pagexml` library. Reads regions in reading order and writes one `.txt` file per page to `outputs/page_text_by_page/`.

---

## Notebooks

All classifier notebooks share the same structure: annotation loading → majority-vote labelling → dossier-level train/val/test split → feature extraction (with caching) → model training and evaluation → per-class metrics summary.

### `page_start_classifier.ipynb`
**Task**: binary classification — is this page the start of a new document segment?  
**Scope**: all annotated pages across 65 dossiers.  
**Models (12 total)**:

| Modality | Models |
|---|---|
| Visual only | KNN+VGG-16, XGBoost+VGG-16, VGG-16 fine-tuned, EfficientNet-B0 fine-tuned, LSTM+VGG-16 |
| Text only | KNN+BERT, XGBoost+BERT, TEXT-CNN, BERT fine-tuned |
| Multimodal | KNN-Ensemble (VGG-16+BERT), XGBoost-Ensemble, Early Fusion (EfficientNet-B0+BERT→MLP), Late Fusion (avg softmax) |

### `layout_start_page_classifier.ipynb`
**Task**: 6-class layout-type classification (Cover, Structured Form, Letter, Card, Photo, Other).  
**Scope**: start pages only (first page of each document segment).  
**Models**: same 12-model grid as above; features are extracted and cached separately (suffix `_layout_start`).

### `layout_full_doc_classifier.ipynb`
**Task**: same 6-class layout-type classification.  
**Scope**: all annotated pages (not just start pages), using the full-document feature cache.  
**Models**: same 12-model grid.

### `function_start_page_classifier.ipynb`
**Task**: 7-class functional-category classification (Application Documents, Decision Documents, Administrative & Internal Processing, Medical & Health, Security & Political Screening, Qualification & Employment Proof, Other).  
**Strategy**: classify start pages only, then propagate each predicted label forward to subsequent pages until the next segment boundary.  
**Models**: same 12-model grid.

### `function_full_doc_classifier.ipynb`
**Task**: same 7-class functional-category classification.  
**Scope**: all annotated pages classified independently (no propagation).  
**Models**: same 12-model grid.

---

## Data

| Path | Contents |
|---|---|
| `NAMA_digitised_page_files/` | Raw PageXML files, one sub-folder per dossier |
| `pdf_pages_png/` | Page-level PNG images rendered by `pdf_to_png.py` |
| `outputs/page_text_by_page/` | Per-page `.txt` files produced by `xml_extraction.py` |
| `annotation 1–4.xlsx` | Four independent human annotations (65 dossiers, ~5 shared) |
| `feature_cache/` | Pre-computed feature matrices keyed by modality and scope |
