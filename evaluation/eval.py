# -*- coding: utf-8 -*-

from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge


def eval(gts, res, metrics='all', args=None):

    all = (metrics == 'all')
    results = dict()

    if ('bleu' in metrics) or all:
        scorer = Bleu(n=4)
        s1, _ = scorer.compute_score(gts, res)
        results['bleu'] = s1
    
    if ('cider' in metrics) or all:
        scorer = Cider()
        s2, _ = scorer.compute_score(gts, res)
        results['cider'] = s2

    if ('meteor' in metrics) or all:
        scorer = Meteor()
        s3, _ = scorer.compute_score(gts, res)
        results['meteor'] = s3

    if ('rouge' in metrics) or all:
        scorer = Rouge()
        s4, _ = scorer.compute_score(gts, res)
        results['rouge'] = s4
          
    return results

def get_bleu(gts,res):
    scorer = Bleu(n=4)
    s, _ = scorer.compute_score(gts, res)
    return s

def get_meteor(gts, res):
    scorer = Meteor()
    s, _ = scorer.compute_score(gts, res)
    return s

def get_cider(gts, res):
    scorer = Cider()
    s, _ = scorer.compute_score(gts, res)
    return s

def get_rouge(gts, res):
    scorer = Rouge()
    s, _ = scorer.compute_score(gts, res)
    return s
