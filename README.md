# Phase1-B — Diabetes Regression (PyTorch MLP)

A compact regression project using the scikit-learn **Diabetes** dataset.  
Includes: proper train/val/test split, EarlyStopping, ReduceLROnPlateau, MAE/MSE tracking, best-checkpoint saving, and an optional Gradio UI.

## Tech
- PyTorch (MLP), scikit-learn, matplotlib, Gradio
- Device: Apple Silicon MPS or CPU

## Files
- `phase1_diabetes_train.py` — training + evaluation + plots
- `gradio_diabetes_app.py` — inference UI (optional)
- `environment.yml` — reproducible env
- `best_diabetes_model.pt` / `scaler_diabetes.joblib` — saved artifacts

## How to run
```bash
conda env create -f environment.yml
conda activate phase1-diabetes-mlp
python phase1_diabetes_train.py
# optional
python gradio_diabetes_app.py

> I just published a new mini-project: **Diabetes Regression with PyTorch MLP**.  
> It features proper train/val/test splits, EarlyStopping, LR scheduling, and a lightweight Gradio app for interactive inference.  
> Dataset: scikit-learn diabetes.  
> Repo: <your GitHub link>  
> (Built with guidance from GPT-5 Thinking while I iterated and learned hands-on.)
