GPT-2 Sentiment Classifier

A Streamlit app that classifies movie/product reviews as Positive or Negative using GPT-2.
Currently supports two modes:
1. Zero-shot (Seq2Seq) – GPT-2 is prompted to answer “yes/no” without fine-tuning.
2. Fine-tuned Classifier – GPT-2 with a classification head trained on a labeled dataset.


Installation

git clone https://github.com/ezhang3333/GPT-2.git
pip install all packages from requirements.txt
python -m streamlit run app.py
