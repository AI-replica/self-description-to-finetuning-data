import json
import anthropic
import time
import os


# Your fact and the generated question will be translated to these languages. 
# For the complete list of languges in the TinyLlama dataset, see:
# https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/README.md
LANGUAGES_LIST = ["Spanish", "Russian"]

PATH_TO_FACTS = "self_facts.txt"

OUTPUT_FILE_PATH = "finetuning_data.json"

USE_ONLY_FIRST_N_FACTS = None  # useful for testing. Set to None to use all facts.

api_key = os.getenv("ANTHROPIC_API_KEY")


CLIENTS = {}


def get_client(api_key=None):
    """Retrieve an anthropic client, caching clients by api_key."""
    key = api_key or "default"
    if key not in CLIENTS:
        CLIENTS[key] = anthropic.Anthropic(api_key=api_key)
    return CLIENTS[key]


def get_response(
    conversation,
    conversation_with_metadata,
    model,
    api_key=None,
    mock7=False,
    system_prompt="You're a helpful assistant",
    N=10,
):
    """Get response from Claude API"""
    report = dict()

    if mock7:
        user_message = conversation[-1]["content"][0]["text"]
        res = f"User said: {user_message}"
    else:
        attempt = 0
        backoff_time = 1  # initial wait time in seconds
        while attempt < N:
            try:
                # Retrieve or create a client with the provided api_key
                client = get_client(api_key)

                response = client.messages.create(
                    model=model,
                    max_tokens=5000,
                    temperature=0.8,
                    system=system_prompt,
                    messages=conversation,
                )
                res = response.content[0].text
                # print(f"Got a respsone at attempt {attempt}")
                break  # Exit loop if successful

            except Exception as e:
                print(f"Error: {e}")
                attempt += 1
                if attempt < N:
                    print(f"Waiting {backoff_time} seconds before retrying...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Increase wait time exponentially
                else:
                    msg = f"Error after {N} attempts: {e}"
                    report["error"] = msg
                    res = None
    return res, report


def ask_ai_for_answer(question):
    conversation = [{"role": "user", "content": question}]

    answer, report = get_response(
        conversation=conversation,
        conversation_with_metadata=[],
        model="claude-3-5-sonnet-latest",
        api_key=api_key,
        mock7=False,
        system_prompt="You're a helpful assistant",
    )
    return answer


def read_facts_from_file(facts_file_path):
    """
    Reads facts from 'structured_self_facts.txt' located in the project directory.

    Returns:
        List[str]: A list containing each fact as a string.
    """

    with open(facts_file_path, "r") as file:
        facts = [line.strip() for line in file if line.strip()]

    # remove lines that start with "<", as they are tags
    facts = [fact for fact in facts if not fact.startswith("<")]

    # if a line starts with "- ", remove the "- "
    for i, fact in enumerate(facts):
        if fact.startswith("- "):
            facts[i] = fact[2:]

    # remove empty strings
    facts = [fact for fact in facts if fact]

    if USE_ONLY_FIRST_N_FACTS is not None:
        facts = facts[:USE_ONLY_FIRST_N_FACTS]

    print(f"Got {len(facts)} facts from {PATH_TO_FACTS}")

    return facts


def get_text_between_tags(text, tag):
    """
    E.g. <question>What is your name?</question>
    """
    sucess7 = False
    try:
        res = text.split(f"<{tag}>")[1].split(f"</{tag}>")[0]
        sucess7 = True
    except Exception as e:
        print(f"Error getting text between tags {tag} in {text}")
        res = None
    return res, sucess7


def convert_fact_to_question(fact):
    prompt = f"""
    Our goal is to build a personality questionaire, step by step.
    We are building it from a list of facts about a person.
    For example we convert "my name is Alexey" to "What is your name?"

    The question should be an open question.
    It should be exactly one question (don't make it a question with multiple parts).
    The question should be generic enough so any well-educated person can answer it. For example, don't ask about a specific pet named Rex mentioned in the fact, but ask about pets in general.

    Please convert the following fact into a question. Return only the question. 
    {fact}

    Question:
    """
    question = ask_ai_for_answer(prompt)
    return question


def translate_question_answer_pair(question, answer, lang, max_attempts=10):
    attempt = 0
    translated_q = None
    translated_a = None
    success7 = False
    already_in_target_language7 = False

    while attempt < max_attempts:
        prompt = f"""
        Translate the following question and answer into {lang} language.
        Question: {question}
        Answer: {answer}

        Return them in the following format:
        <question>here goes the translated question</question>
        <answer>here goes the translated answer</answer>

        If they are already in the target language, just return <already>.

        If for some reason you absolutely must refuse to translate, return <refuse>, and then your explanation.
        Refuse only in exceptionally severe cases, as it's very important to translate everything. Don't be judgemental of peoples' personal opinions. 

        Translated question and answer:
        """
        translation = ask_ai_for_answer(prompt)

        possible_keyword = translation.strip().lower()
        # Check if the translation indicates it's already in the target language
        if possible_keyword.startswith("<already>"):
            already_in_target_language7 = True
            q_excerpt = question[:10]
            print(f"        The text is already in {lang} language: {q_excerpt}...")

            translated_q = question
            translated_a = answer
            success7 = True
            break
        elif possible_keyword.startswith("<refuse>"):
            print(f"        The assistant refused to translate the question and answer into {lang}. Retrying...")
            attempt += 1
            continue  # Retry without breaking the loop
        else:
            translated_q, success_q7 = get_text_between_tags(translation, "question")
            translated_a, success_a7 = get_text_between_tags(translation, "answer")

            if success_q7 and success_a7:
                success7 = True
                if attempt > 0:
                    print(f"        Managed to get a successful translation after {attempt} attempts.")
                break
            else:
                attempt += 1

    if not success7:
        print(f"        Failed to get valid translation after {attempt} attempts. Skipping it.")
        translated_q = None
        translated_a = None
        already_in_target_language7 = False

    return translated_q, translated_a, already_in_target_language7


def generate_dialogs(facts, languages_list):
    """
    Generate dialogs for each fact and each language in languages_list.
    """
    dialogs = []
    counter = 0
    total = len(facts)
    for fact in facts:
        print(f"\nGenerating dialogs for fact #{counter} of {total}: {fact[:50]}...")
        counter += 1
        
        print("    converting to a question...")
        question = convert_fact_to_question(fact)
        
        if question is None:
            print("    Failed to convert to a question. Skipping it.")
            continue

        print(f"        Q: {question}")
        dialogs.append((question, fact))
            
        for lang in languages_list:

            print(f"    translating to {lang}...")
            translated_q, translated_a, already_in_target_language7 = (
                translate_question_answer_pair(question, fact, lang)
            )
            if (translated_a is not None) and (not already_in_target_language7):
                dialogs.append((translated_q, translated_a))
    return dialogs


def generate_json_in_alpaca_like_format(dialogs, output_file_path):
    """
    Generates a JSON file in the Alpaca format from the provided dialogs.

    Args:
        dialogs (List[Tuple[str, str]]): A list of (question, answer) tuples.
        output_file_path (str): The path where the JSON file will be saved.

    The format looks like this:

    [
        {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "1. Eat a balanced diet and make sure to include plenty of fruits and vegetables.\n2. Exercise regularly to keep your body active and strong.\n3. Get enough sleep and maintain a consistent sleep schedule."
        },
        {
            "instruction": "What are the three primary colors?",
            "input": "",
            "output": "The three primary colors are red, blue, and yellow."
        },
        ...
    ]

    See the example here:
    https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
    """

    alpaca_data = []

    for question, answer in dialogs:
        item = {"instruction": question, "input": "", "output": answer}
        alpaca_data.append(item)

    # Write the data to a JSON file
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(alpaca_data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    facts = read_facts_from_file(PATH_TO_FACTS)
    # facts = facts[280:290]
    dialogs = generate_dialogs(facts, LANGUAGES_LIST)
    generate_json_in_alpaca_like_format(dialogs, OUTPUT_FILE_PATH)

    print(f"Finetuning data has been generated and saved to {OUTPUT_FILE_PATH}.")
