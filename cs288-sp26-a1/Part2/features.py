from collections import ChainMap
from typing import Callable, Dict, Set

import pandas as pd


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        # TODO: implement this! Expected # of lines: <5
        words = set(text.lower().split()) - self.STOP_WORDS
        return self.prefix_with_name({w: 1.0 for w in words})


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)


class Bigrams(FeatureMap):
    name = "bigram"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = [w for w in text.lower().split() if w not in self.STOP_WORDS]
        bigrams = set()
        for i in range(len(words) - 1):
            bigrams.add(f"{words[i]}_{words[i+1]}")
        return self.prefix_with_name({b: 1.0 for b in bigrams})


class Trigrams(FeatureMap):
    name = "trigram"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = [w for w in text.lower().split() if w not in self.STOP_WORDS]
        trigrams = set()
        for i in range(len(words) - 2):
            trigrams.add(f"{words[i]}_{words[i+1]}_{words[i+2]}")
        return self.prefix_with_name({t: 1.0 for t in trigrams})


class NegationBigrams(FeatureMap):
    """Captures words that follow negation words — key for sentiment."""
    name = "neg"
    NEGATION_WORDS = {"not", "no", "never", "n't", "neither", "nor", "nobody",
                      "nothing", "nowhere", "don't", "doesn't", "didn't",
                      "won't", "wouldn't", "couldn't", "shouldn't", "can't",
                      "isn't", "aren't", "wasn't", "weren't", "haven't",
                      "hasn't", "hadn't"}

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = text.lower().split()
        feats = set()
        for i in range(len(words) - 1):
            if words[i] in self.NEGATION_WORDS:
                feats.add(f"NOT_{words[i+1]}")
        return self.prefix_with_name({f: 1.0 for f in feats})


class Punctuation(FeatureMap):
    """Punctuation and style features."""
    name = "punct"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        feats = {}
        if "!" in text:
            feats["has_exclaim"] = float(text.count("!"))
        if "?" in text:
            feats["has_question"] = float(text.count("?"))
        if "..." in text or ".." in text:
            feats["has_ellipsis"] = 1.0
        # All-caps words (excluding single chars)
        caps_count = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        if caps_count:
            feats["caps_words"] = float(caps_count)
        return self.prefix_with_name(feats)


class WordSuffix(FeatureMap):
    """Word suffix features — captures part-of-speech patterns."""
    name = "suffix"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = [w for w in text.lower().split() if w not in self.STOP_WORDS and len(w) > 3]
        suffixes = set()
        for w in words:
            suffixes.add(w[-3:])
        return self.prefix_with_name({s: 1.0 for s in suffixes})


FEATURE_CLASSES_MAP = {c.name: c for c in [
    BagOfWords, SentenceLength, Bigrams, Trigrams,
    NegationBigrams, Punctuation, WordSuffix,
]}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
