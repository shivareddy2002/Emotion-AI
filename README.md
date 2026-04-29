<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:6a11cb,100:2575fc&height=180&section=header&text=Emotion%20AI%20-%20Decode%20Human%20Sentiment&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35" />
</p>

A professional deep learning web application that detects human emotions from text in real time using a **Bidirectional LSTM (BiLSTM)** neural network. Built with **TensorFlow, Keras, Flask, and Bootstrap**, this project transforms raw text into meaningful emotional insights.

---


## 🧭 Project Elaboration (Quick Walkthrough)

Emotion-AI is a **production-oriented sentiment understanding app** that combines an NLP model with a Flask UI. In practice, the system works as a pipeline:

1. **User enters free-form text** in the web page (`templates/index.html`).  
2. **Backend sanitizes text** (lowercasing + regex cleanup) before inference (`app.py`).  
3. **Inference strategy is selected at runtime**:  
   - **Local model mode** (TensorFlow assets loaded from disk) for full deep-learning predictions.  
   - **Remote inference mode** (API forwarding) for lightweight cloud deployment such as Vercel limits.  
   - **Lexicon fallback mode** for resilient predictions when ML dependencies are unavailable.
4. **Emotion + confidence** are returned as JSON and rendered instantly in the UI.

### Why this architecture matters
- **Reliability**: graceful fallback prevents total outage if model/runtime is unavailable.  
- **Deployability**: remote inference keeps serverless package size small.  
- **Maintainability**: model assets and app logic are separated (`model_loader.py` vs `app.py`).

### Core repository layout
- `app.py` → Flask routes, runtime mode switching, normalization, and fallback logic.  
- `model_loader.py` → loading tokenizer/encoder/model artifacts.  
- `templates/` + `static/` → web experience (UI, styles, interactions).  
- `requirements*.txt` → dependency sets for local vs deployment scenarios.

## 🚀 Features

- 🔮 Real-time emotion prediction  
- 🧠 BiLSTM-powered deep learning model  
- 🎭 Supports 7 emotions:  
  😊 **Joy** | 😢 **Sadness** | 😡 **Anger** | 😨 **Fear** | 💙 **Love** | 😲 **Surprise** | 😐 **Neutral**

- 🌐 Interactive Flask web interface  


---

## 🏗️ Technical Architecture

1️⃣ **Data Loading & Exploration**  
Dataset is ingested and split into **80% training / 20% testing**.

2️⃣ **Text Cleaning & Tokenization**  
Regex cleaning → Tokenization → Padding to fixed sequence length.

3️⃣ **Model Architecture**  
Embedding → **Bidirectional LSTM** → Dropout → Softmax Output.

4️⃣ **Training & Validation**  
Categorical Cross-Entropy + Adam Optimizer with Early Stopping & Checkpointing.

5️⃣ **Evaluation & Optimization**  
Accuracy, Precision, Recall, Confusion Matrix & Hyperparameter tuning.

6️⃣ **Deployment & Inference**  
Trained model (`emotion_model.h5`) integrated into Flask for real-time prediction.


## 🖥️ Web Interface

- Input text via a modern UI  
- One-click “Try a Sample” emotion buttons  
- Live prediction display with confidence bar  
- Fully responsive design  

---

## 🛠️ Tech Stack

| Layer        | Technology |
|-------------|------------|
| Frontend    | HTML, CSS, JS, Bootstrap |
| Backend     | Flask (Python) |
| ML / DL     | TensorFlow, Keras |
| NLP         | Tokenizer, Padding, Regex |
| Deployment  | Flask Web App |

---

## ⚙️ Workflow & Steps  

### 1️⃣ Data Loading & Preprocessing
- Dataset loaded from `.csv` / `.txt` file  
- Text tokenized using **Keras Tokenizer**  
- Sequences padded for uniform input length  

### 2️⃣ Label Encoding
- Emotion labels converted to numeric form using `LabelEncoder`  
- One-hot encoding applied for multi-class classification  

### 3️⃣ Model Architecture
Built using **Keras Sequential API**:

- 🔹 Embedding Layer – Converts words into dense vectors  
- 🔹 Flatten Layer – Converts embeddings into 1D vector  
- 🔹 Dense Layer – Learns complex patterns  
- 🔹 Output Layer – Softmax activation for multi-class prediction  

Compiled with:
- Optimizer: `Adam`  
- Loss Function: `categorical_crossentropy`  

### 4️⃣ Model Training
- Dataset split using `train_test_split`  
- Trained for **10 epochs**  
- Batch size: **32**  
- Validation data used to monitor performance  

### 5️⃣ Prediction & Testing
- Model predicts emotions for unseen text  
- Example:  
  > Input: `"I am feeling very nostalgic"`  
  > Output: `"Love"`

---
## 🖼️ Visual Workflow

```mermaid
flowchart LR
    subgraph DP[📂 Data Preparation]
        A["📦 Import Libraries"]
        B["📚 Load Dataset"]
        C["✂️ Preprocessing"]
    end

    subgraph LP[🏷️ Label Encoding]
        D["🔢 Encode Emotions"]
        E["📊 One-Hot Encoding"]
    end

    subgraph MT[🤖 Modeling & Training]
        F["🏗️ Build Model"]
        G["⚡ Train Model"]
    end

    subgraph PR[🔮 Prediction]
        H["📝 User Input"]
        I["🔍 Tokenize & Pad"]
        J["🎯 Predict Emotion"]
    end

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
    %% --- Styles ---
    style A fill:#FFD54F,stroke:#F57F17,stroke-width:2px,color:#000;
    style B fill:#4FC3F7,stroke:#0277BD,stroke-width:2px,color:#fff;
    style C fill:#AED581,stroke:#33691E,stroke-width:2px,color:#000;
    style D fill:#FFCC80,stroke:#EF6C00,stroke-width:2px,color:#000;
    style E fill:#FFE082,stroke:#F9A825,stroke-width:2px,color:#000;
    style F fill:#BA68C8,stroke:#4A148C,stroke-width:2px,color:#fff;
    style G fill:#FF8A65,stroke:#BF360C,stroke-width:2px,color:#fff;
    style H fill:#81D4FA,stroke:#01579B,stroke-width:2px,color:#000;
    style I fill:#B3E5FC,stroke:#0288D1,stroke-width:2px,color:#000;
    style J fill:#90CAF9,stroke:#0D47A1,stroke-width:2px,color:#000;
```
## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/shivareddy2002/emotion-ai.git
cd emotion-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## ✨ Key Features

- Text Tokenization & Padding  
- Multi-class Emotion Classification  
- Deep Learning with Embeddings  
- One-Hot Encoded Labels  
- Validation-Based Training  

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Scikit-learn  
- Jupyter Notebook  

---

## 🚀 Applications

- 📱 Social Media Sentiment Analysis  
- 🛒 Customer Feedback Classification  
- 🛡️ Content Moderation  
- 🧠 Mental Health Monitoring  
- 💬 Chatbots & Virtual Assistants  

---

## 🧩 Conclusion  

This project demonstrates how **Neural Networks** can effectively understand and classify human emotions from text.  
It highlights the power of NLP in real-world applications such as customer service, analytics, and well-being platforms.

---

## 👨‍💻 Author  

**Lomada Siva Gangi Reddy**  
- 🎓 B.Tech CSE (Data Science), RGMCET (2021–2025)  
- 💡 Interests: Python | Machine Learning | Deep Learning | Data Science  
- 📍 Open to **Internships & Job Offers**

 **Contact Me**:  

- 📧 **Email**: lomadasivagangireddy3@gmail.com  
- 📞 **Phone**: 9346493592  
- 💼 [LinkedIn](https://www.linkedin.com/in/lomada-siva-gangi-reddy-a64197280/)  🌐 [GitHub](https://github.com/shivareddy2002)  🚀 [Portfolio](https://lsgr-portfolio-pulse.lovable.app/)

---
<!-- Footer Banner -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:2575fc,100:6a11cb&height=120&section=footer"/>
</p>


## ☁️ Deploy on Vercel Without "Project Size Exceeded"

Vercel serverless functions have strict bundle limits, and `tensorflow-cpu` is usually too large.

This repository now supports two modes:

1. **Local mode**: uses TensorFlow model files directly.
2. **Vercel mode (default on Vercel)**: skips TensorFlow and forwards inference requests to a remote API.

### Steps

1. Deploy a small Python API for model inference on a platform that supports heavy ML dependencies (Render/Railway/EC2).
2. In Vercel project settings, add:
   - `USE_LOCAL_MODEL=false`
   - `REMOTE_INFERENCE_URL=https://your-ml-api.example.com/predict`
3. Deploy this repo to Vercel. It installs only `requirements-vercel.txt`, which avoids large ML packages.

### Local development

Use local dependencies when running model inference in the same process:

```bash
pip install -r requirements-local.txt
python app.py
```

### Troubleshooting deployed crashes

If Vercel shows `FUNCTION_INVOCATION_FAILED`:

1. Open `https://<your-domain>/health` and verify:
   - `use_local_model` is `false` on Vercel
   - `remote_inference_configured` is `true`
2. In Vercel Project Settings → Environment Variables, set:
   - `USE_LOCAL_MODEL=false`
   - `REMOTE_INFERENCE_URL=https://your-ml-api.example.com/predict`
3. Redeploy after saving env vars.


### Analyze button shows no results?

If `/predict` cannot reach your remote inference API, the app now returns a lightweight fallback prediction so the UI still responds.
For production-quality predictions, configure `REMOTE_INFERENCE_URL` to your ML service and keep `USE_LOCAL_MODEL=false` on Vercel.


Remote API payload should include either `{emotion, confidence}` or `{label, score}`. If payload is invalid, the app automatically falls back to lightweight local keyword prediction.



### Get best prediction quality

For highest quality predictions in production:

1. Prefer a dedicated remote inference service with your trained TensorFlow model.
2. Set `REMOTE_INFERENCE_URL` in Vercel to that endpoint.
3. Keep `USE_LOCAL_MODEL=false` on Vercel to avoid serverless crashes and size limits.
4. Use `/health` to verify runtime status (`remote_inference_configured=true`).

If remote service is unavailable, the app will safely fall back to the built-in lite model.

### If Vercel still shows `FUNCTION_INVOCATION_FAILED`

1. Ensure `vercel.json` includes `templates/**` and `static/**` in `includeFiles` for `@vercel/python`.
2. Keep `USE_LOCAL_MODEL=false` on Vercel unless TensorFlow and model files are actually available.
3. Check `https://<your-domain>/health` after deploy; it should return JSON (not crash page).
4. If root route fails, verify templates are deployed (missing templates cause serverless 500s).



### Why preview branch works but `main` fails on Vercel

A common root cause is **different Vercel environment configuration between Preview and Production** after merge:

- Preview deployment may have `REMOTE_INFERENCE_URL` configured while Production does not.
- Production may still have stale `USE_LOCAL_MODEL=true` from older deploys.
- Build packaging can differ if `vercel.json` changes were not merged cleanly.

After merging to `main`, verify Production env values in Vercel Project Settings and redeploy Production.

