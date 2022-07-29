import argparse
import html
from typing import List

import pandas as pd
import requests

import json

from nltk.tokenize import TreebankWordTokenizer

API_KEY = ""


def get_translation_with_post(session: requests.Session,
                              source_language: str,
                              target_language: str,
                              utterances: List,
                              return_list: bool = False):
    """
    :desc: translate list of text into target language
    :param session: a session object
    :param source_language: ISO-639-1 Code representing target language to be translated to. Full list => https://cloud.google.com/translate/docs/languages
    :param target_language: ISO-639-1 Code representing target language to be translated to. Full list => https://cloud.google.com/translate/docs/languages
    :param utterances: a list of utterances to be translated
    :param return_list: whether to return as a list
    :return: a dataframe with translatedText column and detectedLanguage column
    """
    url = f'https://translation.googleapis.com/language/translate/v2?key={API_KEY}'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    request_body = {
        "q": utterances,
        "source": source_language,
        "target": target_language
    }
    r = session.post(url=url, json=request_body, headers=headers)
    results = r.json()
    results = results['data']['translations']
    if return_list:
        return [html.unescape(result['translatedText']) for result in results]
    df = pd.DataFrame(results)
    df['translatedText'] = df['translatedText'].apply(lambda x: html.unescape(x))
    return df


def tokenize_translation(text: str) -> List:
    return TreebankWordTokenizer().tokenize(text)


def translate_text(session: requests.Session, text: str, source_language: str, target_language: str):
    return get_translation_with_post(session=session,
                                     source_language=source_language,
                                     target_language=target_language,
                                     utterances=[text],
                                     return_list=True)[0]


def augment_bt_paraphrases_to_spider(spider_input_file: str,
                                     bt_aug_output_file: str,
                                     src_lang: str = "en",
                                     intermediate_lang: str = "de"):
    session = requests.Session()
    new_data = []
    with open(spider_input_file) as finp:
        data = json.load(finp)
        for example in data:
            question = example["question"]

            print(f"Generate paraphrase for question: {question}")

            # translate from English to German
            ende_translation = translate_text(session=session, text=question,
                                              source_language=src_lang,
                                              target_language=intermediate_lang)

            # translate back from German to English
            question_paraphrase = translate_text(session=session, text=ende_translation,
                                                 source_language=intermediate_lang,
                                                 target_language=src_lang)

            if question_paraphrase != question:
                example['question'] = question_paraphrase
                example['question_toks'] = tokenize_translation(text=question_paraphrase)

                print(f"Generated paraphrase: {question_paraphrase}")

                new_data.append(example)

    with open(bt_aug_output_file, 'w') as fout:
        json.dump(obj=new_data, fp=fout, indent=4)


if __name__ == '__main__':
    """
    Usage: python ./scripts/data_aug/bt_para_augment.py --google-api-key <API_KEY> --spider-input-file <SPIDER_INPUT_FILE> --bt-aug-output-file <BT_AUG_OUTPUT_FILE> --src-lang en --intermediate-lang de
    """
    parser = argparse.ArgumentParser(
        description='Generate paraphrase for Spider data based on back translation')
    parser.add_argument("--google-api-key",
                        help="The API KEY to use Google Translate.")
    parser.add_argument("--spider-input-file",
                        help="The Spider input file.")
    parser.add_argument("--bt-aug-output-file",
                        help="The Spider input file.")
    parser.add_argument("--src-lang",
                        help="The source language.",
                        default="en")
    parser.add_argument("--intermediate-lang",
                        help="The target language.",
                        default="de")

    args, _ = parser.parse_known_args()

    API_KEY = args.google_api_key
    augment_bt_paraphrases_to_spider(spider_input_file=args.spider_input_file,
                                     bt_aug_output_file=args.bt_aug_output_file,
                                     src_lang=args.src_lang,
                                     intermediate_lang=args.intermediate_lang)
