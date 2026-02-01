"""
Anchor-based pairing of conditional clauses to main clauses for English learner text.

Pipeline:
1) Constituency (benepar) finds conditional SBAR spans for markers: if/unless/even if
2) For each conditional span, pick a main-clause ANCHOR = nearest finite verb/modal OUTSIDE the conditional
3) MAIN span = local segment around the anchor bounded by punctuation (commas/semicolons/periods),
   with any overlap with the conditional removed.

  
Copyright (c) Daniel Meszaros & Ionela Manda
"""

import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import spacy
import benepar
import stanza
from spacy.tokens import Span

import pandas as pd
import streamlit as st
import io
import zipfile
import altair as alt
import subprocess


@st.cache_resource
def download_en_core_web_sm():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

@dataclass
class Clause:
    """
    Represents a clause in a sentence with grammatical features extracted.
    """
    text: str
    tenses: Set[str] = field(default_factory=set)
    aspects: Set[str] = field(default_factory=set)
    modals: Set[str] = field(default_factory=set)
    verb_forms: Set[str] = field(default_factory=set)
    lemmas: Set[str] = field(default_factory=set)
    auxiliaries: Set[str] = field(default_factory=set)
    moods: Set[str] = field(default_factory=set)
    voices: Set[str] = field(default_factory=set)
    perfect: bool = False

    def analyze(self, nlp_pipeline):
        """
        Analyzes the clause text to extract grammatical features.
        """
        doc = nlp_pipeline(self.text)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in {'VERB', 'AUX'}:
                    self.lemmas.add(word.lemma.lower())
                    if word.upos == 'AUX':
                        self.auxiliaries.add(word.lemma.lower())
                        # Check for modals
                        if word.lemma.lower() in {
                            'will', 'would', 'shall', 'should',
                            'can', 'could', 'may', 'might', 'must'
                        }:
                            self.modals.add(word.lemma.lower())
                    if word.feats:
                        feats = dict(item.split('=') for item in word.feats.split('|'))
                        verb_form = feats.get('VerbForm', '')
                        if 'Tense' in feats and verb_form == 'Fin':
                            self.tenses.add(feats['Tense'])
                        if 'Aspect' in feats:
                            self.aspects.add(feats['Aspect'])
                        if verb_form:
                            self.verb_forms.add(verb_form)
                        if 'Mood' in feats:
                            self.moods.add(feats['Mood'])
                        if 'Voice' in feats:
                            self.voices.add(feats['Voice'])
        # Determine if clause has perfect aspect
        if 'have' in self.auxiliaries and 'Part' in self.verb_forms:
            self.perfect = True

# -------------------- markers --------------------
COND_MARKERS_1 = {"if", "unless"}
COND_MARKERS_2 = {("even", "if")}  # multiword

INVERSION_AUX = {"had", "were", "should"}
DASHES = {"—", "–", "-"}

# -------------------- cleanup --------------------
_PUNCT_SPACES = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACES = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = s.strip()
    s = _PUNCT_SPACES.sub(r"\1", s)
    s = _MULTI_SPACES.sub(" ", s)
    return s.strip(" ,;")

# -------------------- spaCy + benepar --------------------
@st.cache_resource
def build_nlp():
    nlp = spacy.load("en_core_web_sm")
    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    return nlp

def get_sent_tree(sent: Span):
    if Span.has_extension("parse_tree"):
        t = sent._.parse_tree
        if t is not None:
            return t
    if Span.has_extension("parse_string"):
        ps = sent._.parse_string
        if ps:
            from nltk import Tree
            return Tree.fromstring(ps)
    raise RuntimeError("No benepar parse found on sentence.")

def iter_subtrees_with_positions(tree):
    yield tree, ()
    for pos in tree.treepositions():
        node = tree[pos]
        if hasattr(node, "label"):
            yield node, pos

def compute_leaf_spans(tree) -> Dict[Tuple[int, ...], Tuple[int, int]]:
    leaf_positions = list(tree.treepositions("leaves"))
    spans: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    for subtree, pos in iter_subtrees_with_positions(tree):
        if not hasattr(subtree, "label"):
            continue
        idxs = [i for i, lp in enumerate(leaf_positions) if lp[: len(pos)] == pos]
        if idxs:
            spans[pos] = (idxs[0], idxs[-1] + 1)
    return spans

def first_tokens(sent: Span, start: int, end: int, k: int = 4) -> List[str]:
    return [sent[i].text.lower() for i in range(start, min(end, start + k))]

def is_conditional_sbar(tree, pos, spans, sent: Span) -> Optional[str]:
    node = tree[pos]
    if node.label() != "SBAR":
        return None
    sp = spans.get(pos)
    if not sp:
        return None
    start, end = sp
    toks = first_tokens(sent, start, end, k=4)
    if len(toks) >= 2 and (toks[0], toks[1]) in COND_MARKERS_2:
        return "even if"
    if toks and toks[0] in COND_MARKERS_1:
        return toks[0]
    return None

# -------------------- trimming noisy learner spans --------------------
FINITE_TAGS = {"VBD", "VBP", "VBZ", "MD"}  # fast finite-ish cues

def looks_like_new_clause_start(tok) -> bool:
    return tok.lower_ in {"i","you","he","she","it","we","they","there"} or tok.pos_ in {"PRON", "PROPN", "NOUN"}

def has_finite_near(tokens, i, window=5) -> bool:
    for j in range(i, min(len(tokens), i + window)):
        if tokens[j].tag_ in FINITE_TAGS:
            return True
    return False

def trim_conditional_span(doc, sent: Span, doc_start: int, doc_end: int) -> Tuple[int,int]:
    """
    If SBAR overreaches and includes a following independent clause:
      even if X, there's Y
    cut at the comma before the new clause.
    """
    toks = list(doc[sent.start:sent.end])
    s0 = max(0, doc_start - sent.start)
    s1 = min(len(toks), doc_end - sent.start)

    for i in range(s0 + 1, s1 - 1):
        if toks[i].text == ",":
            nxt = toks[i + 1]
            if looks_like_new_clause_start(nxt) and has_finite_near(toks, i + 1):
                new_end = sent.start + i
                if new_end > doc_start + 2:
                    return (doc_start, new_end)
    return (doc_start, doc_end)

# -------------------- extract conditional spans --------------------
def extract_conditional_spans(doc) -> List[Tuple[int, int, str]]:
    """
    Returns list of (doc_token_start, doc_token_end, marker) token spans, end exclusive.
    """
    out = []
    for sent in doc.sents:
        tree = get_sent_tree(sent)
        spans = compute_leaf_spans(tree)

        for subtree, pos in iter_subtrees_with_positions(tree):
            if not hasattr(subtree, "label") or subtree.label() != "SBAR":
                continue
            marker = is_conditional_sbar(tree, pos, spans, sent)
            if not marker:
                continue

            s0, s1 = spans[pos]
            doc_start = sent.start + s0
            doc_end = sent.start + s1

            doc_start, doc_end = trim_conditional_span(doc, sent, doc_start, doc_end)
            out.append((doc_start, doc_end, marker))
    return out

def dedupe_spans(spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    """
    Remove duplicates and spans fully contained in another span with the same marker.
    Keep longer span when nested.
    """
    spans = list({(a,b,m) for a,b,m in spans})
    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))

    kept = []
    for a,b,m in spans:
        contained = False
        for aa,bb,mm in kept:
            if m == mm and aa <= a and b <= bb:
                contained = True
                break
        if not contained:
            kept.append((a,b,m))
    return kept

# -------------------- anchor-based main clause extraction --------------------
BOUNDARY_PUNCT = {",", ";", ":", ".", "!", "?"}

def is_finiteish(tok) -> bool:
    # Penn tags are the most reliable here for English
    return tok.tag_ in FINITE_TAGS or ("VerbForm=Fin" in tok.morph)

def is_modal(tok) -> bool:
    return tok.tag_ == "MD" and tok.lower_ in {"will","would","can","could","may","might","should","must","shall"}

def in_span(i: int, span: Tuple[int,int]) -> bool:
    return span[0] <= i < span[1]

def pick_anchor(doc, sent: Span, cond_span: Tuple[int,int]) -> Optional[int]:
    """
    Choose a finite verb/modal outside cond span.
    Preference:
    - modal (should/would/can/...) outside cond
    - otherwise nearest finite-ish token outside cond
    Direction heuristic:
    - if cond is early in sentence -> prefer to the right
    - if cond is late -> prefer to the left
    - otherwise nearest overall
    """
    s0, s1 = sent.start, sent.end
    cond0, cond1 = cond_span

    candidates = []
    for i in range(s0, s1):
        if in_span(i, cond_span):
            continue
        tok = doc[i]
        if tok.pos_ in {"VERB", "AUX"} and is_finiteish(tok):
            candidates.append(i)

    if not candidates:
        # fallback: any verb/aux outside conditional
        for i in range(s0, s1):
            if in_span(i, cond_span):
                continue
            tok = doc[i]
            if tok.pos_ in {"VERB", "AUX"}:
                candidates.append(i)

    if not candidates:
        return None

    # modal preference
    modals = [i for i in candidates if is_modal(doc[i])]
    if modals:
        # pick the modal closest to conditional boundary
        return min(modals, key=lambda i: min(abs(i - cond0), abs(i - cond1)))

    # direction preference
    sent_len = s1 - s0
    rel_cond0 = cond0 - s0
    rel_cond1 = cond1 - s0

    prefer_right = rel_cond0 <= sent_len * 0.35
    prefer_left = rel_cond1 >= sent_len * 0.70

    if prefer_right:
        right = [i for i in candidates if i >= cond1]
        if right:
            return min(right, key=lambda i: i - cond1)
    if prefer_left:
        left = [i for i in candidates if i < cond0]
        if left:
            return min(left, key=lambda i: cond0 - i)

    # otherwise nearest to conditional midpoint
    mid = (cond0 + cond1) / 2.0
    return min(candidates, key=lambda i: abs(i - mid))

def expand_segment_around_anchor(doc, sent: Span, anchor_i: int) -> Tuple[int,int]:
    """
    Return a sentence-local segment around anchor bounded by punctuation.
    (start,end) in doc token indices, end exclusive.
    """
    s0, s1 = sent.start, sent.end
    start = anchor_i
    end = anchor_i + 1

    # expand left to previous boundary punct (but do not include it)
    i = anchor_i - 1
    while i >= s0:
        if doc[i].text in BOUNDARY_PUNCT:
            break
        start = i
        i -= 1

    # expand right to next boundary punct (stop before it)
    i = anchor_i
    while i < s1:
        if i != anchor_i and doc[i].text in BOUNDARY_PUNCT:
            break
        end = i + 1
        i += 1

    return (start, end)

def remove_overlap(main_span: Tuple[int,int], cond_span: Tuple[int,int]) -> List[Tuple[int,int]]:
    """
    main minus cond => list of 0-2 spans.
    """
    a,b = main_span
    c,d = cond_span
    out = []
    # left remainder
    if a < c:
        out.append((a, min(b, c)))
    # right remainder
    if d < b:
        out.append((max(a, d), b))
    return [(x,y) for x,y in out if y-x > 0]

def spans_to_text(doc, spans: List[Tuple[int,int]]) -> str:
    if not spans:
        return ""
    txt = " ".join(doc[a:b].text for a,b in spans)
    return clean_text(txt)

def pair_conditionals_anchor(text: str):
    nlp = build_nlp()
    doc = nlp(text)

    conds = extract_conditional_spans(doc)
    conds += extract_inversion_conditionals(doc)
    conds = dedupe_spans(conds)

    pairs = []

    for (c0, c1, marker) in conds:
        # find containing sentence
        sent = None
        for s in doc.sents:
            if s.start <= c0 < s.end:
                sent = s
                break
        if sent is None:
            continue

        cond_text = clean_text(doc[c0:c1].text)

        anchor = pick_anchor(doc, sent, (c0, c1))
        if anchor is None:
            continue

        main_seg = expand_segment_around_anchor(doc, sent, anchor)
        main_parts = remove_overlap(main_seg, (c0, c1))
        main_text = spans_to_text(doc, main_parts)

        # If main got too short, widen by one punctuation-bounded segment on each side (still simple)
        if len(main_text.split()) < 3:
            # widen: take from prev punctuation to next punctuation around anchor across the whole sentence
            main_parts = remove_overlap((sent.start, sent.end), (c0, c1))
            main_text = spans_to_text(doc, main_parts)

        pairs.append((marker, cond_text, main_text))

    return pairs
    
    
def is_probable_subject(tok) -> bool:
    # Simple: pronouns and nouns/proper nouns
    return tok.pos_ in {"PRON", "NOUN", "PROPN"}

def is_question_sentence(sent: Span) -> bool:
    # crude but effective
    return any(t.text == "?" for t in sent)

def find_inversion_end(doc, sent: Span) -> int:
    """
    Return doc token index (end-exclusive) for the inverted conditional clause,
    starting at sent.start. Prefer comma/semicolon/dash boundary.
    """
    s0, s1 = sent.start, sent.end

    # Prefer punctuation boundary like: "Had you known, ..."
    for i in range(s0, s1):
        if doc[i].text in {",", ";", ":"} or doc[i].text in DASHES:
            # end at punctuation (exclude it)
            if i > s0 + 1:
                return i

    # Fallback: stop before next boundary punctuation if any
    for i in range(s0 + 2, s1):
        if doc[i].text in {".", "!", "?"}:
            return i

    return s1  # whole sentence fallback

def extract_inversion_conditionals(doc) -> List[Tuple[int,int,str]]:
    """
    Detect clause-initial inverted conditionals:
      Had/Were/Should + Subject ...
    Returns (doc_start, doc_end, marker="inversion").
    """
    out = []
    for sent in doc.sents:
        if is_question_sentence(sent):
            continue

        # Skip leading punctuation/quotes
        i = sent.start
        while i < sent.end and doc[i].is_punct:
            i += 1
        if i + 1 >= sent.end:
            continue

        aux = doc[i].lower_
        subj = doc[i + 1]

        if aux in INVERSION_AUX and is_probable_subject(subj):
            end = find_inversion_end(doc, sent)
            # Ensure the span looks clause-like lengthwise
            if end - i >= 3:
                out.append((i, end, "inversion"))

    return out
    
    
def determine_conditional_type(if_clause: Clause, main_clause: Clause) -> str:
    """
    Determines the type of conditional based on clause analyses.
    """
    if is_zero_conditional(if_clause, main_clause):
        return 'Zero Conditional'
    elif is_first_conditional(if_clause, main_clause):
        return 'First Conditional'
    elif is_second_conditional(if_clause, main_clause):
        return 'Second Conditional'
    elif is_third_conditional(if_clause, main_clause):
        return 'Third Conditional'
    elif is_mixed_conditional(if_clause, main_clause):
        return 'Mixed Conditional'
    else:
        return 'Unknown Conditional'


def is_present_simple(clause: Clause) -> bool:
    """
    Checks if the clause is in the present simple tense.
    """
    return 'Pres' in clause.tenses and not clause.perfect


def is_past_simple(clause: Clause) -> bool:
    """
    Checks if the clause is in the past simple tense.
    """
    return ('Past' in clause.tenses or 'Subj' in clause.moods or 'were' in clause.lemmas) and not clause.perfect


def is_past_perfect(clause: Clause) -> bool:
    """
    Checks if the clause is in the past perfect tense.
    """
    return ('Past' in clause.tenses and clause.perfect) or \
           ('had' in clause.auxiliaries and 'Part' in clause.verb_forms)


def is_present_passive_participle(clause: Clause) -> bool:
    """
    Checks if the clause is a present passive participle.
    """
    return 'Part' in clause.verb_forms and 'Pass' in clause.voices


def has_auxiliary(clause: Clause, auxiliary: str) -> bool:
    """
    Checks if the clause contains a specific auxiliary verb.
    """
    return auxiliary in clause.auxiliaries


def is_imperative(clause: Clause) -> bool:
    """
    Checks if the clause is in the imperative mood.
    """
    return 'Imp' in clause.moods


def has_modal(clause: Clause, modals: Set[str]) -> bool:
    """
    Checks if the clause contains any of the specified modals.
    """
    return bool(clause.modals.intersection(modals))


def is_zero_conditional(if_clause: Clause, main_clause: Clause) -> bool:
    """
    Determines if the condition is a Zero Conditional.
    """
    acceptable_modals = {'can', 'may', 'might', 'must'}
    if_clause_is_present_simple = is_present_simple(if_clause) or has_modal(if_clause, acceptable_modals)
    main_clause_is_present_simple = is_present_simple(main_clause) or has_modal(main_clause, acceptable_modals)
    return if_clause_is_present_simple and \
           main_clause_is_present_simple and \
           not has_modal(main_clause, {'will', 'would', 'should'})



def is_first_conditional(if_clause: Clause, main_clause: Clause) -> bool:
    acceptable_modals = {'will', 'shall', 'can', 'may', 'might', 'could', 'should', 'would'}
    # print("Checking if first conditional")
    # print(f"Main clause: {main_clause.text}")
    # print(f"Conditional clause: {if_clause.text}")
    # print(f"Present Simple: {is_present_simple(if_clause)}")
    return (is_present_simple(if_clause) or has_modal(if_clause, {'should'})) and \
           (has_modal(main_clause, acceptable_modals) or is_imperative(main_clause))



def is_second_conditional(if_clause: Clause, main_clause: Clause) -> bool:
    """
    Determines if the condition is a Second Conditional.
    """
    return is_past_simple(if_clause) and \
           has_modal(main_clause, {'would', 'could', 'might'}) and \
           not has_auxiliary(main_clause, 'have')


def is_third_conditional(if_clause: Clause, main_clause: Clause) -> bool:
    """
    Determines if the condition is a Third Conditional.
    """
    return is_past_perfect(if_clause) and \
           has_modal(main_clause, {'would', 'could', 'might', 'should'}) and \
           has_auxiliary(main_clause, 'have')


def is_mixed_conditional(if_clause: Clause, main_clause: Clause) -> bool:
    """
    Determines if the condition is a Mixed Conditional.
    """
    if_clause_is_past = is_past_simple(if_clause) or is_past_perfect(if_clause)
    main_clause_has_would_have = has_modal(main_clause, {'would', 'could', 'might', 'should'}) and \
                                 has_auxiliary(main_clause, 'have')
    return if_clause_is_past and main_clause_has_would_have



def analyze_texts(texts: List[Tuple[str, str]]) -> pd.DataFrame:
    nlp = get_nlp()

    conditional_markers = {
        "if", "unless", "provided", "provided that", "as long as", "even if",
        "supposing", "assuming", "in case", "on condition that",
        "only if", "in the event that", "even though", "when",
        "suppose", "given that"
    }

    rows = []
    for filename, text in texts:
        doc = nlp(text)
        for sent in doc.sentences:
            pairs = pair_conditionals_anchor(sent.text)
            if not pairs:
                continue
            for (marker, cond, main) in pairs:
                print(f"Cond: {cond}")
                print(f"Main: {main}")
                if_clause = Clause(cond)
                main_clause = Clause(main)
                if_clause.analyze(nlp)
                main_clause.analyze(nlp)
                ctype = determine_conditional_type(if_clause, main_clause)
                rows.append(
                    {
                        "Filename": filename,
                        "Sentence": sent.text,
                        "If clause": cond,
                        "Main clause": main,
                        "Conditional Type": ctype,
                        "Marker": marker
                    }
                )

    return pd.DataFrame(rows)
    
# ----------------------------
# Streamlit helpers
# ----------------------------

@st.cache_resource
def get_nlp():
    #stanza.download("en", verbose=False)
    return stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,lemma,depparse',
        use_gpu=False,
        verbose=False,
        download_method='REUSE_RESOURCES'
    )


def read_uploaded_files(files) -> List[Tuple[str, str]]:
    """Return list of (filename, text). Supports .txt files and a .zip containing .txt files."""
    out: List[Tuple[str, str]] = []
    for f in files:
        name = f.name
        if name.lower().endswith(".zip"):
            zdata = io.BytesIO(f.read())
            with zipfile.ZipFile(zdata) as z:
                for zi in z.infolist():
                    if zi.filename.lower().endswith(".txt") and not zi.is_dir():
                        raw = z.read(zi.filename)
                        text = raw.decode("utf-8", errors="replace")
                        out.append((zi.filename, text))
        elif name.lower().endswith(".txt"):
            text = f.read().decode("utf-8", errors="replace")
            out.append((name, text))
    return out

# -------------------- CLI --------------------
# def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("text", help="Text to analyze.")
    # args = ap.parse_args()

    # pairs = pair_conditionals_anchor(args.text)
    # if not pairs:
        # print("(no conditional pairs found)")
        # return
        
    # nlp = stanza.Pipeline(
        # lang='en',
        # processors='tokenize,pos,lemma,depparse',
        # use_gpu=False,
        # verbose=False,
        # download_method='REUSE_RESOURCES'
    # )

    # for i, (marker, cond, main) in enumerate(pairs, 1):
        # print(f"\nPair {i}")
        # print(f"MARKER: {marker}")
        # print(f"IF:     {cond}")
        # print(f"MAIN:   {main}")
        
        # conditional_clause = Clause(cond)
        # main_clause = Clause(main)
        # conditional_clause.analyze(nlp)
        # main_clause.analyze(nlp)
        
        # conditional_type = determine_conditional_type(conditional_clause, main_clause)
        # print(f"TYPE:   {conditional_type}")
    
    
# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Conditional Classifier", layout="wide")
st.title("Conditional Classifier")
st.caption("Upload .txt files (or a .zip of .txt files) to classify conditionals and visualize results.")

st.markdown(
    """
    <style>
      .credits-card {
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(6px);
      }

      .credits-title {
        font-size: 0.95rem;
        font-weight: 300;
        margin-bottom: 10px;
        color: rgba(255, 255, 255, 0.85);
      }

      .credits-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
      }

      .credits-row:first-of-type {
        border-top: none;
      }

      .credits-name {
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
      }

      .credits-link a {
        text-decoration: none;
        font-weight: 600;
        color: #7ab7ff;
      }

      .credits-link a:hover {
        text-decoration: underline;
      }

      .credits-badges {
        margin-top: 10px;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
      }

      .credits-badge {
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: rgba(255, 255, 255, 0.8);
      }
    </style>
    """,
    unsafe_allow_html=True,
)



with st.sidebar:
    st.header("Input")
    uploads = st.file_uploader(
        "Upload one or more .txt files, or a .zip containing .txt files",
        type=["txt", "zip"],
        accept_multiple_files=True,
    )
    st.divider()
    st.header("Display")
    show_unknown = st.checkbox("Include 'Unknown Conditional'", value=True)
    st.divider()
    
    # <div class="credits-badges">
            # <span class="credits-badge">Streamlit</span>
            # <span class="credits-badge">Stanza</span>
            # <span class="credits-badge">NLP</span>
          # </div>
    
    st.markdown(
        """
        <div class="credits-card">
          <div class="credits-title">Made by</div>

          <div class="credits-row">
            <div class="credits-name">Ionela Tatiana Manda</div>
            <div class="credits-link">
              <a href="https://github.com/ionelatatianamanda-spec" target="_blank">GitHub ↗</a>
            </div>
          </div>

          <div class="credits-row">
            <div class="credits-name">Daniel Meszaros</div>
            <div class="credits-link">
              <a href="https://github.com/meszaros-vasile-daniel-30031" target="_blank">GitHub ↗</a>
            </div>
          </div>

          
        </div>
        """,
        unsafe_allow_html=True,
    )

if not uploads:
    st.info("Upload at least one .txt file (or a .zip) to begin.")
    st.stop()

texts = read_uploaded_files(uploads)
if not texts:
    st.error("No .txt files found in the uploads.")
    st.stop()

with st.spinner("Running Stanza + classifying clauses..."):
    df = analyze_texts(texts)

if df.empty:
    st.warning("No conditional clauses detected in the uploaded text(s).")
    st.stop()

if not show_unknown:
    df = df[df["Conditional Type"] != "Unknown Conditional"].copy()

# Filters
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    file_filter = st.multiselect("Filter by file", sorted(df["Filename"].unique().tolist()))
with colB:
    type_filter = st.multiselect("Filter by type", sorted(df["Conditional Type"].unique().tolist()))
with colC:
    search = st.text_input("Search in sentence / clauses")

df_view = df.copy()
if file_filter:
    df_view = df_view[df_view["Filename"].isin(file_filter)]
if type_filter:
    df_view = df_view[df_view["Conditional Type"].isin(type_filter)]
if search.strip():
    s = search.strip().lower()
    mask = (
        df_view["Sentence"].str.lower().str.contains(s, na=False)
        | df_view["If clause"].str.lower().str.contains(s, na=False)
        | df_view["Main clause"].str.lower().str.contains(s, na=False)
    )
    df_view = df_view[mask]

st.subheader("Charts")

# Chart 1: overall distribution
counts = df_view["Conditional Type"].value_counts().reset_index()
counts.columns = ["Conditional Type", "Count"]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Overall distribution**")
    #st.bar_chart(counts, x="Conditional Type", y="Count", horizontal=True, height='stretch')
    st.altair_chart(alt.Chart(counts).mark_arc(innerRadius=50).encode(
    theta="Count",
    color="Conditional Type:N",
))

with c2:
    st.markdown("**Distribution by file (stacked)**")
    pivot = (
        df_view.pivot_table(
            index="Filename",
            columns="Conditional Type",
            values="Sentence",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .set_index("Filename")
    )
    st.bar_chart(pivot, horizontal=True, height='stretch')

st.subheader("Results table")

st.dataframe(
    df_view.sort_values(["Filename", "Conditional Type"]).reset_index(drop=True),
    use_container_width=True,
    height=420,
)

# Download
csv_bytes = df_view.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results as CSV",
    data=csv_bytes,
    file_name="conditional_results.csv",
    mime="text/csv",
)
