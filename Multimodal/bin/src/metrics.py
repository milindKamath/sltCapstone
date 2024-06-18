'''
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import pdb

def get_bleu_score( gt, pt ):
    b1 = sentence_bleu( [gt], pt, weights=(1,0,0,0) )
    b2 = sentence_bleu( [gt], pt, weights=(0.5,0.5,0,0) )
    b3 = sentence_bleu( [gt], pt, weights=(0.33,0.33,0.33,0) )
    b4 = sentence_bleu( [gt], pt, weights=(0.25,0.25,0.25,0) )
    return b1, b2, b3, b4

def bleu( gts, pts ):
    b1, b2, b3, b4 = 0, 0, 0, 0
    count = 0
    for i in range( len(gts) ):
        gt = gts[i]
        pt = pts[i]
        s1, s2, s3, s4 = get_bleu_score( gt, pt )
        b1 += s1
        b2 += s2
        b3 += s3
        b4 += s4
        count += 1
    b1 = round( ( b1 / count ) * 100, 6 )   #(12.967479674796753, 4.098360655737702, 1.6528925619834673, 0.8333333333333328)
    b2 = round( ( b2 / count ) * 100, 6 )
    b3 = round( ( b3 / count ) * 100, 6 )
    b4 = round( ( b4 / count ) * 100, 6 )

    scores = {
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'b4': b4
    }
    return scores
'''
from collections import Counter
from itertools import zip_longest
from typing import List, Union, Iterable
import math

import pdb


NGRAM_ORDER = 4
SMOOTH_VALUE_DEFAULT = 0.0
TOKENIZERS = {
    "none": lambda x: x,
}

def my_log(num):
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)

class BLEU:
    def __init__(self, scores, counts, totals, precisions, bp, sys_len, ref_len):

        self.scores = scores
        self.counts = counts
        self.totals = totals
        self.precisions = precisions
        self.bp = bp
        self.sys_len = sys_len
        self.ref_len = ref_len

    def format(self, width=2):
        precisions = "/".join(["{:.1f}".format(p) for p in self.precisions])
        return "BLEU = {scores} {precisions} (BP = {bp:.3f} ratio = {ratio:.3f} hyp_len = {sys_len:d} ref_len = {ref_len:d})".format(
            scores=self.scores,
            width=width,
            precisions=precisions,
            bp=self.bp,
            ratio=self.sys_len / self.ref_len,
            sys_len=self.sys_len,
            ref_len=self.ref_len,
        )


def compute_bleu(
    correct: List[int],
    total: List[int],
    sys_len: int,
    ref_len: int,
    smooth_method="none",
    smooth_value=SMOOTH_VALUE_DEFAULT,
    use_effective_order=False,
) -> BLEU:
    """Computes BLEU score from its sufficient statistics. Adds smoothing.

    Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
    Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

    - exp: NIST smoothing method (Method 3)
    - floor: Method 1
    - add-k: Method 2 (generalizing Lin and Och, 2004)
    - none: do nothing.

    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param sys_len: The cumulative system length
    :param ref_len: The cumulative reference length
    :param smooth: The smoothing method to use
    :param smooth_value: The smoothing value added, if smooth method 'floor' is used
    :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of NGRAM_ORDER.
    :return: A BLEU object with the score (100-based) and other statistics.
    """
    # pdb.set_trace()
    precisions = [0 for x in range(NGRAM_ORDER)]

    smooth_mteval = 1.0
    effective_order = NGRAM_ORDER
    for n in range(NGRAM_ORDER):
        if smooth_method == "add-k" and n > 1:
            correct[n] += smooth_value
            total[n] += smooth_value
        if total[n] == 0:
            break

        if use_effective_order:
            effective_order = n + 1

        if correct[n] == 0:
            if smooth_method == "exp":
                smooth_mteval *= 2
                precisions[n] = 100.0 / (smooth_mteval * total[n])
            elif smooth_method == "floor":
                precisions[n] = 100.0 * smooth_value / total[n]
        else:
            precisions[n] = 100.0 * correct[n] / total[n]

    # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU score is 0 (technically undefined).
    # This is a problem for sentence-level BLEU or a corpus of short sentences, where systems will get no credit
    # if sentence lengths fall under the NGRAM_ORDER threshold. This fix scales NGRAM_ORDER to the observed
    # maximum order. It is only available through the API and off by default

    brevity_penalty = 1.0
    if sys_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

    scores = []
    for effective_order in range(1, NGRAM_ORDER + 1):
        scores.append(
            brevity_penalty
            * math.exp(sum(map(my_log, precisions[:effective_order])) / effective_order)
        )

    return BLEU(scores, correct, total, precisions, brevity_penalty, sys_len, ref_len)

def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
    """Extracts all the ngrams (min_order <= n <= max_order) from a sequence of tokens.

    :param line: A segment containing a sequence of words.
    :param min_order: Minimum n-gram length (default: 1).
    :param max_order: Maximum n-gram length (default: NGRAM_ORDER).
    :return: a dictionary containing ngrams and counts
    """

    ngrams = Counter()
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams[ngram] += 1

    return ngrams


def ref_stats(output, refs):
    ngrams = Counter()
    closest_diff = None
    closest_len = None
    for ref in refs:
        tokens = ref.split()
        reflen = len(tokens)
        diff = abs(len(output.split()) - reflen)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_len = reflen
        elif diff == closest_diff:
            if reflen < closest_len:
                closest_len = reflen

        ngrams_ref = extract_ngrams(ref)
        for ngram in ngrams_ref.keys():
            ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

    return ngrams, closest_diff, closest_len


def corpus_bleu(
    sys_stream: Union[str, Iterable[str]],
    ref_streams: Union[str, List[Iterable[str]]],
    smooth_method="exp",
    smooth_value=SMOOTH_VALUE_DEFAULT,
    force=False,
    lowercase=False,
    tokenize='none',
    use_effective_order=False,
) -> BLEU:
    """Produces BLEU scores along with its sufficient statistics from a source against one or more references.

    :param sys_stream: The system stream (a sequence of segments)
    :param ref_streams: A list of one or more reference streams (each a sequence of segments)
    :param smooth: The smoothing method to use
    :param smooth_value: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :return: a BLEU object containing everything you'd want
    """
    # pdb.set_trace()
    # Add some robustness to the input arguments
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    sys_len = 0
    ref_len = 0

    correct = [0 for n in range(NGRAM_ORDER)]
    total = [0 for n in range(NGRAM_ORDER)]

    # look for already-tokenized sentences
    tokenized_count = 0
    sys_stream = [ ' '.join(s) for s in sys_stream ]
    ref_streams = [ ' '.join(s) for s in ref_streams ]

    fhs = [sys_stream] + [ref_streams]
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        if not (force or tokenize == "none") and lines[0].rstrip().endswith(" ."):
            tokenized_count += 1

            # if tokenized_count == 100:
            #     logging.warning("That's 100 lines that end in a tokenized period ('.')")
            #     logging.warning(
            #         "It looks like you forgot to detokenize your test data, which may hurt your score."
            #     )
            #     logging.warning(
            #         "If you insist your data is detokenized, or don't care, you can suppress this message with '--force'."
            #     )

        output, *refs = [ x.rstrip() for x in lines]

        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs)

        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = extract_ngrams(output)
        for ngram in sys_ngrams.keys():
            n = len(ngram.split())
            correct[n - 1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n - 1] += sys_ngrams[ngram]

    return compute_bleu(
        correct,
        total,
        sys_len,
        ref_len,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        use_effective_order=use_effective_order,
    )


def raw_corpus_bleu(sys_stream, ref_streams, smooth_value=SMOOTH_VALUE_DEFAULT) -> BLEU:
    """Convenience function that wraps corpus_bleu().
    This is convenient if you're using sacrebleu as a library, say for scoring on dev.
    It uses no tokenization and 'floor' smoothing, with the floor default to 0 (no smoothing).

    :param sys_stream: the system stream (a sequence of segments)
    :param ref_streams: a list of one or more reference streams (each a sequence of segments)
    """
    return corpus_bleu(
        sys_stream,
        ref_streams,
        smooth_method="floor",
        smooth_value=smooth_value,
        force=True,
        tokenize="none",
        use_effective_order=True,
    )


def bleu(references, hypotheses):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    # pdb.set_trace()
    bleu_scores = raw_corpus_bleu( sys_stream=hypotheses, ref_streams=references ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["b" + str(n + 1)] = bleu_scores[n]
    return scores