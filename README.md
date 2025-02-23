This script generates finetuning data in the [Alpaca format](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) from a list of self-description facts.

The script works as follows:
1. Convert each fact into a question.
2. Translate the question and the answer into several languages (this greatly helps the model to better remember the facts themselves, instead of memorizing the text as a sequence of tokens).
3. Save the original and the translated questions and the answers to json in the Alpaca format.

Example:
1. "My name is Roman" -> "What is your name?"
2. ("What is your name?", "My name is Roman") -> ("¿Cómo te llamas?", "Me llamo Roman")
3. The result is a json file like this:
[
    {
        "instruction": "What is your name?",
        "input": "",
        "output": "Roman"
    },
    {
        "instruction": "¿Cómo te llamas?",
        "input": "",
        "output": "Roman"
    }
]

Installation:

1. Get your Anthropic API key

2. Clone this project to a folder of your choice:

```git clone https://github.com/AI-replica/self-description-to-finetuning-data.git```

3. Install dependencies:
```pip install -r requirements.txt```

4. Set the API key:
```export ANTHROPIC_API_KEY="your_api_key"```

5. If necessary, set the correct path to the facts file.

6. Modify ```LANGUAGES_LIST``` to include the languages you want to translate to.

7. Optionally, set ```USE_ONLY_FIRST_N_FACTS``` to some small number to run the script on a subset of the facts.

8. Run the script:
```python finetuning_data_generator.py```
