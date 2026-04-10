# Sentiment analysis magisterka
A comparative study of Sentiment Analysis on Amazon reviews using Support Vector Machines (SVM), Long Short-Term Memory (LSTM) networks, and BERT transformers.
## Installation & Setup

Clone the repository and run the following bootstrap script to set up the environment:

```bash
# Clone the repository
git clone <your-repo-url>
cd sentiment-analysis-magisterka

# Setup virtual environment
python3 -m venv .venv

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate

# Update core tools and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Link virtual environment to Jupyter
python -m ipykernel install --user --name=sent-mag-env --display-name "Python (Sent-Mag)"