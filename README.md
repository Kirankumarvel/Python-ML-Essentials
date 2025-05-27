# **Python-ML-Essentials** 🚀  
*A curated collection of essential Python libraries for Machine Learning, Deep Learning, NLP, Data Science, MLOps, and Deployment with documentation links and examples.*

## 📌 Repository Overview
This repository serves as a **quick-reference guide** for the most important Python libraries used in:
- Machine Learning
- Deep Learning  
- NLP
- Computer Vision
- Data Science
- MLOps & Deployment
- Data Visualization & Utilities

---

## 📊 Complete Library Reference Table

| Category                | Library                                                                 | Use Case                               | Installation                        | Resources                                                                              |
|-------------------------|-------------------------------------------------------------------------|----------------------------------------|--------------------------------------|----------------------------------------------------------------------------------------|
| **Core ML**             | [Scikit-learn](https://scikit-learn.org)                                | General-purpose ML                     | `pip install scikit-learn`           | [Docs](https://scikit-learn.org/stable/documentation.html) [Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/) |
|                         | [XGBoost](https://xgboost.ai)                                           | Gradient Boosting                      | `pip install xgboost`                | [Docs](https://xgboost.readthedocs.io) [Tutorial](https://xgboost.ai/tutorials)         |
|                         | [LightGBM](https://lightgbm.readthedocs.io)                             | Fast Gradient Boosting                 | `pip install lightgbm`               | [Docs](https://lightgbm.readthedocs.io)                                                |
|                         | [CatBoost](https://catboost.ai)                                         | Boosting with categorical features     | `pip install catboost`               | [Docs](https://catboost.ai/en/docs/)                                                   |
|                         | [MLflow](https://mlflow.org)                                            | Experiment Tracking & Model Registry   | `pip install mlflow`                 | [Docs](https://mlflow.org/docs/latest/index.html)                                      |
| **Deep Learning**       | [TensorFlow](https://www.tensorflow.org)                                | Neural Networks                        | `pip install tensorflow`             | [Docs](https://www.tensorflow.org/api_docs) [Keras Guide](https://keras.io/guides/)     |
|                         | [PyTorch](https://pytorch.org)                                          | Research DL                            | `pip install torch torchvision`      | [Docs](https://pytorch.org/docs) [Tutorials](https://pytorch.org/tutorials/)           |
|                         | [Keras](https://keras.io)                                               | High-level DL API                      | `pip install keras`                  | [Docs](https://keras.io/guides/)                                                       |
|                         | [Hugging Face Transformers](https://huggingface.co/transformers/)       | Pre-trained DL models                  | `pip install transformers`           | [Docs](https://huggingface.co/docs/transformers/index)                                 |
| **NLP**                 | [spaCy](https://spacy.io)                                               | Text Processing                        | `pip install spacy`                  | [Docs](https://spacy.io/usage) [Models](https://spacy.io/models)                       |
|                         | [NLTK](https://www.nltk.org)                                            | Linguistic Processing                  | `pip install nltk`                   | [Docs](https://www.nltk.org)                                                           |
|                         | [Gensim](https://radimrehurek.com/gensim/)                              | Topic Modeling/Word Embeddings         | `pip install gensim`                 | [Docs](https://radimrehurek.com/gensim/)                                               |
|                         | [TextBlob](https://textblob.readthedocs.io/)                            | Simple NLP Tasks                       | `pip install textblob`               | [Docs](https://textblob.readthedocs.io/)                                               |
| **Computer Vision**     | [OpenCV](https://opencv.org)                                            | Image Processing                       | `pip install opencv-python`          | [Docs](https://docs.opencv.org) [Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html) |
|                         | [Pillow](https://python-pillow.org/)                                    | Imaging (PIL)                          | `pip install pillow`                 | [Docs](https://pillow.readthedocs.io/)                                                 |
|                         | [Albumentations](https://albumentations.ai/)                            | Image Augmentation                     | `pip install albumentations`         | [Docs](https://albumentations.ai/docs/)                                                |
|                         | [torchvision](https://pytorch.org/vision/stable/index.html)             | Vision datasets/models (PyTorch)       | `pip install torchvision`            | [Docs](https://pytorch.org/vision/stable/index.html)                                   |
| **Data Processing**     | [Pandas](https://pandas.pydata.org)                                     | Data Analysis                          | `pip install pandas`                 | [Docs](https://pandas.pydata.org/docs) [Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) |
|                         | [NumPy](https://numpy.org/)                                             | Numerical Computing                    | `pip install numpy`                  | [Docs](https://numpy.org/doc/)                                                         |
|                         | [Dask](https://dask.org/)                                               | Parallel Computing                     | `pip install dask`                   | [Docs](https://docs.dask.org/en/stable/)                                               |
|                         | [Polars](https://www.pola.rs/)                                          | Lightning-fast DataFrames              | `pip install polars`                 | [Docs](https://pola-rs.github.io/polars-book/)                                         |
| **Data Visualization**  | [Matplotlib](https://matplotlib.org)                                    | Plotting                               | `pip install matplotlib`              | [Docs](https://matplotlib.org/stable/contents.html) [Gallery](https://matplotlib.org/stable/gallery/index.html) |
|                         | [Seaborn](https://seaborn.pydata.org/)                                 | Statistical Plots                      | `pip install seaborn`                | [Docs](https://seaborn.pydata.org/)                                                    |
|                         | [Plotly](https://plotly.com/python/)                                   | Interactive Plots                      | `pip install plotly`                 | [Docs](https://plotly.com/python/)                                                     |
|                         | [Altair](https://altair-viz.github.io/)                                | Declarative Visualization              | `pip install altair`                 | [Docs](https://altair-viz.github.io/)                                                  |
| **MLOps/Deployment**    | [Streamlit](https://streamlit.io)                                      | Web Apps                               | `pip install streamlit`              | [Docs](https://docs.streamlit.io) [Gallery](https://streamlit.io/gallery)              |
|                         | [Gradio](https://gradio.app/)                                          | Interactive Demos                      | `pip install gradio`                 | [Docs](https://www.gradio.app/docs/)                                                   |
|                         | [FastAPI](https://fastapi.tiangolo.com/)                               | REST API for ML                        | `pip install fastapi[all]`           | [Docs](https://fastapi.tiangolo.com/)                                                  |
|                         | [ONNX](https://onnx.ai/)                                               | Model Interchange                      | `pip install onnx`                   | [Docs](https://onnx.ai/docs/)                                                          |
| **Utilities**           | [joblib](https://joblib.readthedocs.io/)                               | Model Serialization                    | `pip install joblib`                 | [Docs](https://joblib.readthedocs.io/)                                                 |
|                         | [tqdm](https://tqdm.github.io/)                                        | Progress Bars                          | `pip install tqdm`                   | [Docs](https://tqdm.github.io/)                                                        |
|                         | [Hydra](https://hydra.cc/)                                             | Config Management                      | `pip install hydra-core`             | [Docs](https://hydra.cc/docs/intro/)                                                   |

---

## 🛠️ Quick Start
```bash
# Clone repo
git clone https://github.com/Kirankumarvel/Python-ML-Essentials.git

# Install all core libraries (optional)
pip install -r requirements.txt
```

---

## 📚 Learning Resources
1. [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
2. [PyTorch Deep Learning Course](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
3. [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
4. [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
5. [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1)
6. [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
7. [MLOps with MLflow](https://mlflow.org/docs/latest/index.html)

---

## 🤝 Contributing
Contribute by:
1. Adding new libraries
2. Updating documentation links
3. Improving examples

---

## 📜 License
MIT © 2025 | Kiran Kumar
