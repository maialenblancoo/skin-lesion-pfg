# Multimodal Skin Lesion Classification — TFG

Final degree project (TFG) at Universidad de Deusto.

This project develops a multimodal deep learning system for skin lesion
classification on the **HAM10000** dataset, combining dermatoscopic images
(EfficientNet-B0) with clinical metadata (sex, age, anatomical location) through
late fusion and a selective weighted ensemble. Beyond accuracy, it focuses on
clinical safety and trust: a melanoma-oriented decision threshold, explainability
(Grad-CAM, SmoothGrad, SHAP, image-vs-metadata ablation) and predictive-uncertainty
estimation (MC Dropout).

The repository contains the full code (`src/`), the complete experimental workflow
(`notebooks/`, from EDA to the final explainability analysis) and the results
(`outputs/figures`, `outputs/metrics`). The HAM10000 dataset is not included — it can
be downloaded from its [official source](https://doi.org/10.7910/DVN/DBW86T). The two
final trained models are hosted on the
[Hugging Face Hub](https://huggingface.co/maialenblancoo/skin-lesion-pfg).

**Author:** Maialen Blanco Ibarra · Universidad de Deusto, 2026
