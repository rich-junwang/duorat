import abc
from functools import lru_cache
from typing import List, Sequence, Tuple, Optional, Dict

import stanza
from transformers import AutoTokenizer  # , BertTokenizerFast
from transformers import BasicTokenizer

from duorat.utils import registry, corenlp


class AbstractTokenizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def tokenize(self, s: str) -> List[str]:
        pass

    @abc.abstractmethod
    def detokenize(self, xs: Sequence[str]) -> str:
        pass


@registry.register("tokenizer", "CoreNLPTokenizer")
class CoreNLPTokenizer(AbstractTokenizer):
    def __init__(self):
        pass

    @lru_cache(maxsize=1024)
    def _tokenize(self, s: str) -> List[str]:
        ann = corenlp.annotate(
            text=s,
            annotators=["tokenize", "ssplit"],
            properties={
                "outputFormat": "serialized",
                "tokenize.options": "asciiQuotes = false, latexQuotes=false, unicodeQuotes=false, ",
            },
        )
        return [tok.word for sent in ann.sentence for tok in sent.token]

    def tokenize(self, s: str) -> List[str]:
        return [token.lower() for token in self._tokenize(s)]

    def tokenize_with_raw(self, s: str) -> List[Tuple[str, str]]:
        return [(token.lower(), token) for token in self._tokenize(s)]

    def detokenize(self, xs: Sequence[str]) -> str:
        return " ".join(xs)


@registry.register("tokenizer", "StanzaTokenizer")
class StanzaTokenizer(AbstractTokenizer):
    def __init__(self):
        stanza.download("en", processors="tokenize")
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize")

    @lru_cache(maxsize=1024)
    def tokenize(self, s: str) -> List[str]:
        doc = self.nlp(s)
        return [
            token.question for sentence in doc.sentences for token in sentence.tokens
        ]

    def detokenize(self, xs: Sequence[str]) -> str:
        return " ".join(xs)


@registry.register("tokenizer", "BERTTokenizer")
class BERTTokenizer(AbstractTokenizer):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 cls_token: Optional[str] = None,
                 sep_token: Optional[str] = None):
        # self._bert_tokenizer = BertTokenizerFast.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path
        # )
        self._bert_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        if cls_token is not None:
            self._bert_tokenizer.cls_token = cls_token
        if sep_token is not None:
            self._bert_tokenizer.sep_token = sep_token
        self._basic_tokenizer = BasicTokenizer()
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._subword_sep_token = '##'

    def tokenize(self, s: str) -> List[str]:
        return self._bert_tokenizer.tokenize(s)

    def _maybe_lowercase(self, tok: str) -> str:
        if hasattr(self._bert_tokenizer, "do_lower_case") and self._bert_tokenizer.do_lower_case:
            return tok.lower()
        return tok

    @classmethod
    def _preprocess_non_standard_quote(cls, text: str) -> str:
        return text.replace('‘', '\'').replace('’', '\'').replace('“', '"').replace('”', '"').strip()

    def _get_raw_token_strings(self, enc_res: Dict, ori_s: str, toks: List[str]) -> List[str]:
        assert len(enc_res['input_ids']) == len(toks) + 2

        return [
            self._basic_tokenizer._run_strip_accents(ori_s[start:end]) for start, end in
            enc_res["offset_mapping"][1:-1]
        ]

    def _update_raw_token_strings_with_sharps(self, tokens: Tuple[str, str],
                                              raw_token_strings_with_sharps: List[str]):
        token, raw_token = tokens
        assert (
                token == self._maybe_lowercase(raw_token)
                or token[len(self._subword_sep_token):] == self._maybe_lowercase(raw_token)
                or token[-len(self._subword_sep_token):] == self._maybe_lowercase(raw_token)
        )

        if token.startswith(self._subword_sep_token):
            raw_token_strings_with_sharps.append(f"{self._subword_sep_token}{raw_token}")
        elif token.endswith(self._subword_sep_token):
            raw_token_strings_with_sharps.append(f"{raw_token}{self._subword_sep_token}")
        else:
            raw_token_strings_with_sharps.append(raw_token)

    def tokenize_with_raw(self, s: str) -> List[Tuple[str, str]]:
        # TODO: at some point, hopefully, transformers API will be mature enough
        # to do this in 1 call instead of 2
        s = self._preprocess_non_standard_quote(text=s)
        tokens = self._bert_tokenizer.tokenize(s)
        encoding_result = self._bert_tokenizer(s, return_offsets_mapping=True)
        raw_token_strings = self._get_raw_token_strings(enc_res=encoding_result,
                                                        ori_s=s,
                                                        toks=tokens)
        raw_token_strings_with_sharps = []
        for token, raw_token in zip(tokens, raw_token_strings):
            # handle [UNK] token
            if str(token) == self._bert_tokenizer.unk_token:  # '[UNK]':
                raw_token_strings_with_sharps.append(raw_token)
                continue

            self._update_raw_token_strings_with_sharps(tokens=(token, raw_token),
                                                       raw_token_strings_with_sharps=raw_token_strings_with_sharps)

        return zip(tokens, raw_token_strings_with_sharps)

    def detokenize(self, xs: Sequence[str]) -> str:
        """Naive implementation, see https://github.com/huggingface/transformers/issues/36"""
        text = " ".join([x for x in xs])
        fine_text = text.replace(f" {self._subword_sep_token}", "")
        return fine_text

    def convert_token_to_id(self, s: str) -> int:
        return self._bert_tokenizer.convert_tokens_to_ids(s)

    @property
    def cls_token(self) -> str:
        return self._bert_tokenizer.cls_token

    @property
    def sep_token(self) -> str:
        return self._bert_tokenizer.sep_token


@registry.register("tokenizer", "RoBERTaTokenizer")
class RoBERTaTokenizer(BERTTokenizer):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 cls_token: Optional[str] = None,
                 sep_token: Optional[str] = None):
        super(RoBERTaTokenizer, self).__init__(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               cls_token=cls_token,
                                               sep_token=sep_token)
        self._subword_sep_token = 'Ġ'

    def detokenize(self, xs: Sequence[str]) -> str:
        text = "".join([x for x in xs])
        fine_text = text.replace(self._subword_sep_token, " ")
        return fine_text

    def _update_raw_token_strings_with_sharps(self,
                                              tokens: Tuple[str, str],
                                              raw_token_strings_with_sharps: List[str]):
        token, raw_token = tokens
        assert (
                token == self._maybe_lowercase(raw_token)
                or token == self._subword_sep_token
                or token[len(self._subword_sep_token):] == self._maybe_lowercase(raw_token)
        )

        if token == self._subword_sep_token:
            raw_token_strings_with_sharps.append(self._subword_sep_token)
        elif token.startswith(self._subword_sep_token):
            raw_token_strings_with_sharps.append(f"{self._subword_sep_token}{raw_token}")
        else:
            raw_token_strings_with_sharps.append(raw_token)


@registry.register("tokenizer", "T5Tokenizer")
class T5Tokenizer(RoBERTaTokenizer):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 cls_token: Optional[str] = None,
                 sep_token: Optional[str] = None):
        super(T5Tokenizer, self).__init__(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                          cls_token=cls_token,
                                          sep_token=sep_token)
        self._subword_sep_token = '▁'

    def _get_raw_token_strings(self, enc_res: Dict, ori_s: str, toks: List[str]) -> List[str]:
        assert len(enc_res['input_ids']) == len(toks) + 1

        return [
            self._basic_tokenizer._run_strip_accents(ori_s[start:end]) for start, end in
            enc_res["offset_mapping"][:-1]
        ]
