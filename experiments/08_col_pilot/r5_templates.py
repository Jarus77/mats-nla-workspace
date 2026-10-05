"""Round-5 FIXED templates (written before any round-5 scoring). Signs: + = the steered "+" value.

col   (+ = longer line / break sooner): the pilot's fixed col templates (templates.py) PLUS 07's FROZEN
      selected templates (templates_frozen.json, sha256 09156722..., copied verbatim) for nj1_col
      (count) and nj1_col_break (decision), PLUS OL newline-continuation candidates (next token is a
      line break vs another word; the pair differs only by the leading newline).
digit (+ = next item number 7, - = 5): numeral and qualitative families.
tense (+ = Past): 07's FROZEN selected templates (OL continuation forms; NLA register forms) plus the
      calibration's fixed label templates.
"""
import templates as COLT

F07 = {  # verbatim from experiments/07_template_mining/templates_frozen.json ("selected")
    "nla": {
        "f07_count": [("Readout taken late in the span of the wrapped line.", "Readout taken early in the span of the wrapped line."),
                      ("Mid-sentence continuation located near the end of a hard-wrapped line.",
                       "Mid-sentence continuation located at the beginning of a hard-wrapped line."),
                      ("The token appears far into of the wrapped line.", "The token appears near the beginning of the wrapped line.")],
        "f07_break": [("Wrap state: line break coming soon.", "Wrap state: line break far away."),
                      ("Line break status: imminent line break under automatic wrap.",
                       "Line break status: mid-line, far from wrap threshold."),
                      ("Automatic linewrap decision is breaking soon.", "Automatic linewrap decision is not breaking yet.")],
        "f07_tense": [("Historical register", "Descriptive register"), ("Historical factual", "Descriptive factual"),
                      ("Historical tone", "Descriptive tone")],
    },
    "ol": {
        "f07_count": [("- deep after a newline", "- just after a newline"),
                      ("- line position: late in the line", "- line position: early in the line"),
                      ("- current line depth: deep", "- current line depth: shallow")],
        "f07_break": [("- line wrap status: line break imminent", "- line wrap status: mid-line, no break yet"),
                      ("- near linewrap boundary", "- far from linewrap boundary"),
                      ("- line break is coming soon", "- line break is far away")],
        "f07_tense": [("- They were", "- They are"), ("- The village had", "- The village has"), ("- It was", "- It is")],
    },
}

NL_CONT_OL = [("- \nthe", "- the"), ("- \nand", "- and"), ("- \nof", "- of")]

NLA_PAIRS = {
    "col": {**COLT.NLA_PAIRS, "f07_count": F07["nla"]["f07_count"], "f07_break": F07["nla"]["f07_break"]},
    "digit": {"numeral": [("The next list item will be number 7", "The next list item will be number 5"),
                          ("Numbered list; the next item is item 7", "Numbered list; the next item is item 5")],
              "qualitative": [("Deep into a long numbered list", "Early in a short numbered list")]},
    "tense": {"f07_tense": F07["nla"]["f07_tense"],
              "label": [("Past-tense narrative", "Present-tense narrative"),
                        ("The verb is in the past tense", "The verb is in the present tense"),
                        ("Describes events that happened in the past", "Describes events that are happening now")]},
}
OL_PAIRS = {
    "col": {**COLT.OL_PAIRS, "f07_count": F07["ol"]["f07_count"], "f07_break": F07["ol"]["f07_break"],
            "nl_cont": NL_CONT_OL},
    "digit": {"numeral": [("- 7.", "- 5."), ("- seventh item", "- fifth item")],
              "qualitative": [("- deep into a long numbered list", "- early in a short numbered list")]},
    "tense": {"f07_tense": F07["ol"]["f07_tense"],
              "label": [("- past tense", "- present tense"), ("- happened in the past", "- happening now"),
                        ("- was", "- is")]},
}
JLENS_WORDS = {
    "col": COLT.JLENS_WORDS,
    "digit": {"numeral": [(["7", "seven", "seventh"], ["5", "five", "fifth"])]},
    "tense": {"label": [(["past", "was", "were"], ["present", "is", "are"])],
              "content": [(["was", "were", "had", "did", "said", "became", "went", "made", "took", "came"],
                           ["is", "are", "has", "does", "says", "becomes", "goes", "makes", "takes", "comes"])]},
}
PRIMARY = {"col": ("decision", "f07_break", "nl_cont", "count_qual", "f07_count"),
           "digit": ("numeral", "qualitative"), "tense": ("f07_tense", "label")}


def flat(pairs: dict) -> list[tuple]:
    return [(fam, a, b) for fam, lst in pairs.items() for a, b in lst]
