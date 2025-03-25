# Generative Text Model

This project implements and trains generative text models using both GPT-2 and LSTM architectures. The models generate text based on an input prompt and are trained on a dataset.

## Project Structure
```
GenerativeTextModel/
│-- models/                 # Folder to store trained models
│-- logs/                   # Folder to store training logs
│-- src/
│   │-- generate_text.py    # Script to generate text using GPT-2
│   │-- tokenizer.py        # Script to create and save tokenizer
│   │-- train_gpt.py        # Script to train GPT-2 model
│   │-- train_lstm.py       # Script to train LSTM model
│-- requirements.txt        # Dependencies for the project
```

## Installation
Ensure you have Python installed, then install the required dependencies:
```sh
pip install -r requirements.txt
```

## Training Models

### Train GPT-2 Model
To train the GPT-2 model, run:
```sh
python src/train_gpt.py
```
This will train a GPT-2 model on a subset of the WikiText-2 dataset and save it in `models/gpt_model`.

### Train LSTM Model
To train the LSTM model, run:
```sh
python src/train_lstm.py
```
This will train an LSTM model using a predefined dataset and save it as `models/lstm_model.h5`.

## Generating Text
To generate text using the trained GPT-2 model, run:
```sh
python src/generate_text.py
```
Modify the script to provide different prompts.

## Tokenizer
The tokenizer is created and saved using:
```sh
python src/tokenizer.py
```
This ensures consistency in text processing across models.

## Dependencies
The required dependencies are listed in `requirements.txt`:
- TensorFlow
- NumPy
- Matplotlib
- TensorFlow Hub
- Pillow

## Future Improvements
- Enhance dataset size and variety for better generalization.
- Implement fine-tuning techniques for improved text generation.
- Explore alternative transformer models beyond GPT-2.
