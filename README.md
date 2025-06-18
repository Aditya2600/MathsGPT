# MathsGPT 🧮
A smart Streamlit-based chatbot powered by Groq's Gemma 2 model using LangChain. It solves mathematical and logical problems step-by-step and can fetch data from Wikipedia as well.

## 🚀 Features
- Natural language math problem solver
- Wikipedia-powered question answering
- Streamlit chat interface
- Agent with tools (Calculator, Reasoning, Wikipedia)

## 🧠 Tech Stack
- LangChain
- Groq's Gemma 2 via ChatGroq
- Streamlit
- WikipediaAPIWrapper

## 🛠️ Setup

```bash
git clone https://github.com/Aditya2600/MathsGPT.git
cd MathsGPT
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
streamlit run app.py