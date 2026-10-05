"""FIXED templates for the matched-dose in-J calibration (K3), written BEFORE any calibration
scoring. Sign convention follows 02's steering directions:
  lang : + = Russian, - = Polish   (02 lang direction = class-mean Russian - Polish)
  tense: + = Past,    - = Present  (02 tense direction = class-mean Past - Pres)
  col  : the col templates of templates.py (+ = longer line)
"""
import templates as COLT

NLA_PAIRS = {
    "lang": {"label": [("Russian-language text", "Polish-language text"),
                       ("The text is written in Russian", "The text is written in Polish"),
                       ("A passage in Russian", "A passage in Polish")]},
    "tense": {"label": [("Past-tense narrative", "Present-tense narrative"),
                        ("The verb is in the past tense", "The verb is in the present tense"),
                        ("Describes events that happened in the past", "Describes events that are happening now")]},
    "col": COLT.NLA_PAIRS,
}
OL_PAIRS = {
    "lang": {"label": [("- Russian language", "- Polish language"), ("- Russian", "- Polish"),
                       ("- written in Russian", "- written in Polish")]},
    "tense": {"label": [("- past tense", "- present tense"), ("- happened in the past", "- happening now"),
                        ("- was", "- is")]},
    "col": COLT.OL_PAIRS,
}
JLENS_WORDS = {
    "lang": {"label": [(["Russian", "Russia", "русский"], ["Polish", "Poland", "polski"])]},
    "tense": {"label": [(["past", "was", "were"], ["present", "is", "are"])]},
    "col": COLT.JLENS_WORDS,
}
PRIMARY = {"lang": "label", "tense": "label", "col": "count_qual"}


def flat(pairs: dict) -> list[tuple[str, object, object]]:
    return [(fam, a, b) for fam, lst in pairs.items() for a, b in lst]
