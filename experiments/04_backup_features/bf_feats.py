"""Backup-feature definitions (pure python). All features are binary (class 1 = ``pos_name``).

Priorities (lead, Fable round 4): P1 = NON-count non-J candidates (capitalisation, coding language,
in-quote state, French gender agreement of an upcoming noun); P2 = exactly two within-family count
replications (indentation, words since sentence start); P3 = IN-J count calibration (list index with
the numerals in context, "Item N:"), which should come out enriched and J-Lens-readable.

``label_words``: J-Lens value-naming tokens per class (class 0 words, class 1 words), used exactly
like 02 s3d_functional: score_c = -log10(best rank among class-c tokens + 1) in the standard J-Lens
readout softmax(W_U norm(J h)); statistic = mean one-vs-rest AUROC; null = 200 label permutations.

``behaviour`` (G4 metric at the feature's eval positions; steering +u toward class 1 is predicted
to INCREASE it):
  next_indent          E[indent of the next line | next-token distribution] at newline tokens
  sent_end_logit       logit P(next token is sentence-final punctuation . ! ?)
  next_list_number     E[next digit 1-9] at the token before each "Item N:" index digit
  close_vs_open_quote  log P(closing-quote tokens) - log P(opening-quote tokens ' "')
  fem_vs_masc          log P(feminine noun/adj lexicon) - log P(masculine lexicon) (corpus lexicon)
  cap_upper_vs_lower   log P(" "+Uppercase...) - log P(" "+lowercase...)
  py_vs_js             log P(Python-marker tokens) - log P(JavaScript-marker tokens)
"""

from __future__ import annotations

from dataclasses import dataclass

NUM = {k: (str(k), w) for k, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
     "nineteen", "twenty"])}


def _nums(*ks: int) -> tuple[str, ...]:
    return tuple(x for k in ks for x in NUM[k])


@dataclass(frozen=True)
class BFeature:
    name: str
    dataset: str
    neg_name: str
    pos_name: str
    description: str
    label_words: tuple  # (class-0 words, class-1 words)
    behaviour: str
    role: str  # "P1-noncount" | "P2-count" | "P3-inJ-calibration"
    position_matched: bool  # classes subsampled to equal absolute-position distributions
    question: str  # self-report yes/no-style 2AFC; answers[1] means class 1
    answers: tuple


FEATURES = [
    BFeature(
        "indent", "code_py", "level1", "level2",
        "Indentation level of the current line in Python code (2 vs 1 indents of 4 spaces), read at "
        "mid-line tokens (not the first code token of the line). MBPP + HumanEval solutions.",
        (_nums(1, 4) + ("single", "shallow", "outer", "top"), _nums(2, 8) + ("double", "nested", "inner", "deep")),
        "next_indent", "P2-count", False,
        "Is the last line of this code indented by two levels (8 spaces) rather than one level (4 spaces)?",
        ("No", "Yes"),
    ),
    BFeature(
        "sent_pos", "prose", "early", "late",
        "Words since sentence start at the current word: late (>=16th word) vs early (2nd-5th word); "
        "never a paragraph's first sentence; classes matched on absolute token position.",
        (_nums(2, 3, 4, 5) + ("start", "beginning", "early", "first", "short"),
         _nums(16, 17, 18, 19, 20) + ("end", "ending", "late", "long", "last")),
        "sent_end_logit", "P2-count", True,
        "Is the unfinished last sentence of this text already more than 15 words long?", ("No", "Yes"),
    ),
    BFeature(
        "list_num", "list_num", "early", "late",
        "IN-J CALIBRATION. Index of the current item in a numbered list with the numerals in context "
        "('Item 1: ... Item 2: ...'; wikitext sentences as items): late (5-7) vs early (2-3), read at "
        "mid-item word tokens; position-matched.",
        (_nums(2, 3) + ("second", "third", "early", "first"), _nums(5, 6, 7) + ("fifth", "sixth", "seventh", "late")),
        "next_list_number", "P3-inJ-calibration", True,
        "Is the list item currently being written the fifth item or later?", ("No", "Yes"),
    ),
    BFeature(
        "cap_next", "prose", "lowercase", "capitalised",
        "Is the NEXT word capitalised? Mid-sentence positions only (next word not sentence-initial).",
        (("lowercase", "small", "lower", "common"), ("capital", "Capital", "uppercase", "proper", "Name")),
        "cap_upper_vs_lower", "P1-noncount", False,
        "Will the next word in this text start with a capital letter?", ("No", "Yes"),
    ),
    BFeature(
        "cap_next_all", "prose", "lowercase", "capitalised",
        "Is the NEXT word capitalised? All word boundaries (sentence-initial positives included).",
        (("lowercase", "small", "lower", "common"), ("capital", "Capital", "uppercase", "proper", "Name")),
        "cap_upper_vs_lower", "P1-noncount", False,
        "Will the next word in this text start with a capital letter?", ("No", "Yes"),
    ),
    BFeature(
        "code_lang", "code_pair", "JavaScript", "Python",
        "Language of a HumanEvalPack snippet (Python vs JavaScript, content-matched problems), read at "
        "code tokens of the canonical solution; group = problem (pairs never split).",
        (("JavaScript", "javascript", "js", "JS", "Node", "npm"), ("Python", "python", "py", "Py", "pip", "Django")),
        "py_vs_js", "P1-noncount", False,
        "Is this code written in Python (rather than JavaScript)?", ("No", "Yes"),
    ),
]
FEATURES += [
    BFeature(
        "in_quote", "dialogue", "narration", "inside_quote",
        "Is the current word inside an open double-quote span (character speech) vs narration? "
        "TinyStories; word tokens, excluding the first/last word of each span and quote tokens.",
        (("narration", "narrative", "story", "narrator"), ("dialogue", "speech", "quote", "said", "talking")),
        "close_vs_open_quote", "P1-noncount", False,
        "Is the last word of this text inside a quotation (a character speaking)?", ("No", "Yes"),
    ),
    BFeature(
        "gender_next", "french", "masc", "fem",
        "Grammatical gender of the UPCOMING head noun (Fem vs Masc; spaCy fr_core_news_md), read at a "
        "gender-neutral plural determiner (les/des/ces/ses/aux/...) before the noun appears. French "
        "Wikipedia.",
        (("masculin", "masculine", "male", "il", "le", "un"), ("féminin", "feminine", "female", "elle", "la", "une")),
        "fem_vs_masc", "P1-noncount", False,
        "Will the noun that this unfinished French text is about to name be grammatically feminine?",
        ("No", "Yes"),
    ),
]
FEATURES += [
    BFeature(
        "list_hdr", "list_num", "early", "late",
        "IN-J CALIBRATION (header). Numbered list 'Item N: ...', read at the token right before the "
        "index digit N (the model is about to emit N; previous numerals in context): N in 5-7 vs 2-3; "
        "position-matched.",
        (_nums(2, 3) + ("second", "third", "early", "first"), _nums(5, 6, 7) + ("fifth", "sixth", "seventh", "late")),
        "next_list_number", "P3-inJ-calibration", True,
        "Is the next list item number 5 or higher?", ("No", "Yes"),
    ),
]
FEATURES_BY_NAME = {f.name: f for f in FEATURES}
DATASETS = sorted({f.dataset for f in FEATURES})

PY_MARKERS = [" def", "def", " self", " elif", " None", " True", " False", " not", " and", " or",
              "):\n", ":\n", " lambda", " len", " range", "#", " print"]
JS_MARKERS = [" function", "function", " const", " let", " var", " null", " true", " false", " ===",
              " !==", "){\n", ") {\n", " {\n", "};\n", ";\n", " =>", "//", " console"]
