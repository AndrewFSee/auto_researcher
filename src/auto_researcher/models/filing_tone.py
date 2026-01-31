"""
10-K Filing Tone Change Model
=============================

A quantitative model that analyzes year-over-year tone changes in 10-K filings
to predict future stock returns.

================================================================================
ACADEMIC FOUNDATION
================================================================================

Key Research:
1. Loughran & McDonald (2011) "When Is a Liability Not a Liability?"
   - Created financial-specific sentiment word lists
   - Generic dictionaries (Harvard-IV) misclassify financial text
   - Finance-specific negative words predict returns

2. Li (2010) "The Information Content of Forward-Looking Statements"
   - MD&A tone predicts future earnings and stock returns
   - Positive tone change → +2-3% CAR over next quarter

3. Feldman et al. (2010) "Management's Tone Change"
   - Tone CHANGES matter more than tone LEVELS
   - Year-over-year tone improvement predicts positive returns
   - ~1.5% abnormal return for positive tone changes

4. Bodnaruk, Loughran, McDonald (2015) "Using 10-K Text to Gauge Financial Constraints"
   - Constraining language predicts future returns and distress
   - Firms with more uncertainty words underperform

5. Garcia (2013) "Sentiment During Recessions"
   - Negative sentiment is more predictive in down markets
   - Asymmetric predictive power

Expected Performance:
- Long positive tone change, short negative change: +3-4% annual alpha
- Works best as medium-term signal (3-6 month holding)
- Signal strength peaks 1-2 weeks after filing

================================================================================
SIGNAL CONSTRUCTION
================================================================================

TONE METRICS (using Loughran-McDonald dictionary):
1. Negative Tone = negative_words / total_words
2. Positive Tone = positive_words / total_words  
3. Uncertainty = uncertainty_words / total_words
4. Litigious = litigious_words / total_words
5. Net Tone = (positive - negative) / (positive + negative)
6. Constraining = constraining_words / total_words

TONE CHANGE SIGNAL:
- Compare current 10-K to prior year 10-K
- Calculate delta for each metric
- Composite = weighted sum of changes

Signal Direction:
- BULLISH: Net tone improved significantly (+0.05 or more)
- NEUTRAL: Minimal change (-0.03 to +0.03)
- BEARISH: Net tone worsened significantly (-0.05 or more)

Expected Outcomes (based on academic research):
- Bullish signal: +2-4% excess return over 3 months
- Bearish signal: -2-3% excess return over 3 months

================================================================================
USAGE
================================================================================

    from auto_researcher.models.filing_tone import FilingToneModel
    
    model = FilingToneModel()
    signal = model.get_signal("AAPL")
    
    print(f"Net Tone: {signal.net_tone:.3f}")
    print(f"Tone Change: {signal.tone_change:+.3f}")
    print(f"Signal: {signal.direction}")
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal, List, Dict, Set, Tuple
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


# ==============================================================================
# LOUGHRAN-MCDONALD SENTIMENT DICTIONARIES
# ==============================================================================

# Loughran-McDonald (2011) Negative Words - most predictive for returns
# This is a curated subset of the most impactful negative words
LM_NEGATIVE = {
    'abandon', 'abandoned', 'abandoning', 'abandonment', 'abandons',
    'abdicate', 'abdicated', 'abdicates', 'abdicating', 'abdication',
    'aberrant', 'aberration', 'aberrations',
    'abeyance', 'abeyances',
    'abnormal', 'abnormalities', 'abnormality', 'abnormally',
    'abolish', 'abolished', 'abolishes', 'abolishing', 'abolition',
    'abrupt', 'abruptly',
    'absence', 'absences', 'absent',
    'abuse', 'abused', 'abuser', 'abusers', 'abuses', 'abusing', 'abusive',
    'accident', 'accidental', 'accidentally', 'accidents',
    'accusation', 'accusations', 'accuse', 'accused', 'accuses', 'accusing',
    'acquiesce', 'acquiesced', 'acquiesces', 'acquiescing',
    'adjourn', 'adjourned', 'adjourning', 'adjournment', 'adjourns',
    'adversarial', 'adversaries', 'adversary',
    'adverse', 'adversely', 'adversity',
    'aftermath', 'aftermaths',
    'against',
    'aggravate', 'aggravated', 'aggravates', 'aggravating', 'aggravation',
    'allegation', 'allegations', 'allege', 'alleged', 'allegedly', 'alleges', 'alleging',
    'annul', 'annulled', 'annulling', 'annulment', 'annulments', 'annuls',
    'anomalies', 'anomalous', 'anomalously', 'anomaly',
    'anticompetitive',
    'antitrust',
    'argue', 'argued', 'argues', 'arguing', 'argument', 'argumentative', 'arguments',
    'arrearage', 'arrearages', 'arrears',
    'arrest', 'arrested', 'arrests',
    'assertions',
    'attrition',
    'averse', 'aversion',
    'bad', 'badly',
    'bail', 'bailout',
    'bankrupt', 'bankruptcies', 'bankruptcy', 'bankrupts',
    'bans', 'banned', 'banning',
    'barrier', 'barriers',
    'bottleneck', 'bottlenecks',
    'breach', 'breached', 'breaches', 'breaching',
    'breakdown', 'breakdowns',
    'bribe', 'bribed', 'bribery', 'bribes', 'bribing',
    'burden', 'burdened', 'burdening', 'burdens', 'burdensome',
    'careless', 'carelessly', 'carelessness',
    'catastrophe', 'catastrophes', 'catastrophic', 'catastrophically',
    'caution', 'cautionary', 'cautioned', 'cautioning', 'cautions', 'cautious', 'cautiously',
    'cease', 'ceased', 'ceases', 'ceasing',
    'censure', 'censured', 'censures', 'censuring',
    'challenge', 'challenged', 'challenges', 'challenging',
    'chaos', 'chaotic',
    'circumvent', 'circumvented', 'circumventing', 'circumvention', 'circumvents',
    'claim', 'claimed', 'claiming', 'claims',
    'clawback',
    'close', 'closed', 'closedown', 'closedowns', 'closes', 'closing', 'closings', 'closure', 'closures',
    'collapse', 'collapsed', 'collapses', 'collapsing',
    'collusion', 'collusions',
    'complain', 'complained', 'complaining', 'complains', 'complaint', 'complaints',
    'compulsion', 'compulsory',
    'concern', 'concerned', 'concerning', 'concerns',
    'condemn', 'condemnation', 'condemnations', 'condemned', 'condemning', 'condemns',
    'confess', 'confessed', 'confesses', 'confessing', 'confession', 'confessions',
    'confiscate', 'confiscated', 'confiscates', 'confiscating', 'confiscation', 'confiscations',
    'conflict', 'conflicted', 'conflicting', 'conflicts',
    'confront', 'confrontation', 'confrontational', 'confrontations', 'confronted', 'confronting', 'confronts',
    'confuse', 'confused', 'confuses', 'confusing', 'confusion',
    'conspiracy', 'conspiracies', 'conspire', 'conspired', 'conspires', 'conspiring',
    'contempt',
    'contention', 'contentions', 'contentious', 'contentiously',
    'contingency', 'contingencies', 'contingent',
    'contradict', 'contradicted', 'contradicting', 'contradiction', 'contradictions', 'contradictory', 'contradicts',
    'contrary',
    'controversial', 'controversies', 'controversy',
    'convict', 'convicted', 'convicting', 'conviction', 'convictions', 'convicts',
    'corrected', 'correcting', 'correction', 'corrections', 'corrective',
    'corrupt', 'corrupted', 'corrupting', 'corruption', 'corruptions', 'corrupts',
    'costly', 'costlier', 'costliest',
    'counterclaim', 'counterclaimed', 'counterclaiming', 'counterclaims',
    'counterfeit', 'counterfeited', 'counterfeiter', 'counterfeiters', 'counterfeiting', 'counterfeits',
    'crime', 'crimes', 'criminal', 'criminally', 'criminals',
    'crises', 'crisis',
    'critical', 'critically', 'criticism', 'criticisms', 'criticize', 'criticized', 'criticizes', 'criticizing',
    'crucial',
    'curtail', 'curtailed', 'curtailing', 'curtailment', 'curtailments', 'curtails',
    'cut', 'cutback', 'cutbacks', 'cuts', 'cutting',
    'damage', 'damaged', 'damages', 'damaging',
    'danger', 'dangerous', 'dangerously', 'dangers',
    'deadlock', 'deadlocked', 'deadlocks',
    'death', 'deaths',
    'debarment', 'debarments', 'debarred',
    'deceit', 'deceitful', 'deceive', 'deceived', 'deceives', 'deceiving', 'deception', 'deceptions', 'deceptive', 'deceptively',
    'decline', 'declined', 'declines', 'declining',
    'defamation', 'defamations', 'defamatory', 'defame', 'defamed', 'defames', 'defaming',
    'default', 'defaulted', 'defaulting', 'defaults',
    'defeat', 'defeated', 'defeating', 'defeats',
    'defect', 'defective', 'defects',
    'defendant', 'defendants',
    'defer', 'deficiency', 'deficiencies', 'deficient',
    'deficit', 'deficits',
    'defraud', 'defrauded', 'defrauding', 'defrauds',
    'delay', 'delayed', 'delaying', 'delays',
    'delete', 'deleted', 'deletes', 'deleting', 'deletion', 'deletions',
    'deleterious',
    'delinquencies', 'delinquency', 'delinquent', 'delinquently', 'delinquents',
    'delist', 'delisted', 'delisting', 'delists',
    'demise',
    'demolish', 'demolished', 'demolishes', 'demolishing', 'demolition', 'demolitions',
    'demote', 'demoted', 'demotes', 'demoting', 'demotion', 'demotions',
    'denial', 'denials', 'denied', 'denies', 'deny', 'denying',
    'deplete', 'depleted', 'depletes', 'depleting', 'depletion', 'depletions',
    'depreciation',
    'depress', 'depressed', 'depresses', 'depressing', 'depression', 'depressions',
    'deprivation', 'deprivations', 'deprive', 'deprived', 'deprives', 'depriving',
    'derelict', 'dereliction',
    'deteriorate', 'deteriorated', 'deteriorates', 'deteriorating', 'deterioration', 'deteriorations',
    'detract', 'detracted', 'detracting', 'detraction', 'detracts',
    'detriment', 'detrimental', 'detrimentally', 'detriments',
    'devastate', 'devastated', 'devastates', 'devastating', 'devastation',
    'deviate', 'deviated', 'deviates', 'deviating', 'deviation', 'deviations',
    'difficult', 'difficulties', 'difficulty',
    'diminish', 'diminished', 'diminishes', 'diminishing',
    'disadvantage', 'disadvantaged', 'disadvantageous', 'disadvantages',
    'disaffiliation',
    'disagree', 'disagreed', 'disagreeing', 'disagreement', 'disagreements', 'disagrees',
    'disallow', 'disallowance', 'disallowances', 'disallowed', 'disallowing', 'disallows',
    'disappoint', 'disappointed', 'disappointing', 'disappointingly', 'disappointment', 'disappointments', 'disappoints',
    'disapproval', 'disapprovals', 'disapprove', 'disapproved', 'disapproves', 'disapproving',
    'disaster', 'disasters', 'disastrous', 'disastrously',
    'disclaim', 'disclaimed', 'disclaimer', 'disclaimers', 'disclaiming', 'disclaims',
    'discontinuance', 'discontinuances', 'discontinuation', 'discontinuations', 'discontinue', 'discontinued', 'discontinues', 'discontinuing',
    'discourage', 'discouraged', 'discourages', 'discouraging',
    'discredit', 'discredited', 'discrediting', 'discredits',
    'discrepancies', 'discrepancy',
    'disfavor', 'disfavored', 'disfavoring', 'disfavors',
    'disgorge', 'disgorged', 'disgorgement', 'disgorgements', 'disgorges', 'disgorging',
    'dishonest', 'dishonestly', 'dishonesty',
    'disincentive', 'disincentives',
    'disloyal', 'disloyalty',
    'dismal', 'dismally',
    'dismiss', 'dismissal', 'dismissals', 'dismissed', 'dismisses', 'dismissing',
    'displace', 'displaced', 'displacement', 'displacements', 'displaces', 'displacing',
    'dispose', 'disposed', 'disposes', 'disposing', 'disposition', 'dispositions',
    'dispute', 'disputed', 'disputes', 'disputing',
    'disqualification', 'disqualifications', 'disqualified', 'disqualifies', 'disqualify', 'disqualifying',
    'disregard', 'disregarded', 'disregarding', 'disregards',
    'disrupt', 'disrupted', 'disrupting', 'disruption', 'disruptions', 'disruptive', 'disrupts',
    'dissatisfaction', 'dissatisfied',
    'dissent', 'dissented', 'dissenter', 'dissenters', 'dissenting', 'dissents',
    'dissolution', 'dissolutions',
    'distort', 'distorted', 'distorting', 'distortion', 'distortions', 'distorts',
    'distress', 'distressed', 'distresses', 'distressing',
    'divest', 'divested', 'divesting', 'divestiture', 'divestitures', 'divestment', 'divestments', 'divests',
    'doubt', 'doubted', 'doubtful', 'doubting', 'doubts',
    'downgrade', 'downgraded', 'downgrades', 'downgrading',
    'downturn', 'downturns',
    'downward', 'downwards',
    'drag', 'dragged', 'dragging', 'drags',
    'drain', 'drained', 'draining', 'drains',
    'drop', 'dropped', 'dropping', 'drops',
    'drought', 'droughts',
    'duress',
    'dysfunction', 'dysfunctional', 'dysfunctions',
    'egregious', 'egregiously',
    'embezzle', 'embezzled', 'embezzlement', 'embezzlements', 'embezzler', 'embezzlers', 'embezzles', 'embezzling',
    'encroach', 'encroached', 'encroaches', 'encroaching', 'encroachment', 'encroachments',
    'encumber', 'encumbered', 'encumbering', 'encumbers', 'encumbrance', 'encumbrances',
    'endanger', 'endangered', 'endangering', 'endangerment', 'endangers',
    'erode', 'eroded', 'erodes', 'eroding', 'erosion',
    'err', 'errant', 'erred', 'erring', 'erroneous', 'erroneously', 'error', 'errors', 'errs',
    'escalate', 'escalated', 'escalates', 'escalating', 'escalation', 'escalations',
    'evade', 'evaded', 'evades', 'evading', 'evasion', 'evasions', 'evasive',
    'exacerbate', 'exacerbated', 'exacerbates', 'exacerbating', 'exacerbation', 'exacerbations',
    'exaggerate', 'exaggerated', 'exaggerates', 'exaggerating', 'exaggeration',
    'exception', 'exceptional', 'exceptionally', 'exceptions',
    'excessive', 'excessively',
    'exclusion', 'exclusionary', 'exclusions',
    'exorbitant', 'exorbitantly',
    'exploit', 'exploitation', 'exploitations', 'exploited', 'exploiting', 'exploits',
    'expose', 'exposed', 'exposes', 'exposing', 'exposure', 'exposures',
    'expropriate', 'expropriated', 'expropriates', 'expropriating', 'expropriation', 'expropriations',
    'fail', 'failed', 'failing', 'fails', 'failure', 'failures',
    'fallout',
    'false', 'falsely', 'falsification', 'falsifications', 'falsified', 'falsifies', 'falsify', 'falsifying', 'falsity',
    'fatal', 'fatalities', 'fatality', 'fatally',
    'fault', 'faulted', 'faults', 'faulty',
    'fear', 'feared', 'fearful', 'fearing', 'fears',
    'felony', 'felonies',
    'fine', 'fined', 'fines', 'fining',
    'fire', 'fired', 'fires', 'firing',
    'flaw', 'flawed', 'flaws',
    'flee', 'fleeing', 'flees', 'fled',
    'fluctuate', 'fluctuated', 'fluctuates', 'fluctuating', 'fluctuation', 'fluctuations',
    'forbid', 'forbidden', 'forbidding', 'forbids',
    'force', 'forced', 'forces', 'forcing', 'forclosure', 'forclosures',
    'foreclose', 'foreclosed', 'forecloses', 'foreclosing', 'foreclosure', 'foreclosures',
    'forfeit', 'forfeited', 'forfeiting', 'forfeits', 'forfeiture', 'forfeitures',
    'fraud', 'frauds', 'fraudulent', 'fraudulently',
    'freeze', 'freezes', 'freezing', 'frozen',
    'frustrate', 'frustrated', 'frustrates', 'frustrating', 'frustratingly', 'frustration', 'frustrations',
    'fugitive',
    'grievance', 'grievances',
    'grossly',
    'guilty',
    'halt', 'halted', 'halting', 'halts',
    'hamper', 'hampered', 'hampering', 'hampers',
    'harass', 'harassed', 'harasses', 'harassing', 'harassment',
    'hardship', 'hardships',
    'harm', 'harmed', 'harmful', 'harmfully', 'harming', 'harms',
    'harsh', 'harshest', 'harshly', 'harshness',
    'hazard', 'hazardous', 'hazards',
    'hinder', 'hindered', 'hindering', 'hinders', 'hindrance', 'hindrances',
    'hostile', 'hostilities', 'hostility',
    'hurt', 'hurting', 'hurts',
    'idle', 'idled', 'idling',
    'ignore', 'ignored', 'ignores', 'ignoring',
    'ill', 'illegal', 'illegalities', 'illegality', 'illegally', 'illegible',
    'illicit', 'illicitly',
    'illiquid', 'illiquidity',
    'imbalance', 'imbalances',
    'immaterial',
    'impair', 'impaired', 'impairing', 'impairment', 'impairments', 'impairs',
    'impasse', 'impasses',
    'impede', 'impeded', 'impedes', 'impediment', 'impediments', 'impeding',
    'impending',
    'imperative',
    'imperfection', 'imperfections',
    'implicate', 'implicated', 'implicates', 'implicating',
    'impose', 'imposed', 'imposes', 'imposing', 'imposition', 'impositions',
    'impossibility', 'impossible',
    'impound', 'impounded', 'impounding', 'impounds',
    'impractical',
    'imprecise', 'imprecision',
    'improper', 'improperly',
    'inability',
    'inaccessible',
    'inaccuracies', 'inaccuracy', 'inaccurate', 'inaccurately',
    'inaction', 'inactions',
    'inadequacies', 'inadequacy', 'inadequate', 'inadequately',
    'inadvertent', 'inadvertently',
    'inappropriate', 'inappropriately',
    'incapable', 'incapacitated', 'incapacity',
    'incidence', 'incidences', 'incident', 'incidents',
    'incompatibilities', 'incompatibility', 'incompatible',
    'incompetence', 'incompetency', 'incompetent', 'incompetently', 'incompetents',
    'incomplete', 'incompletely', 'incompleteness',
    'inconsistencies', 'inconsistency', 'inconsistent', 'inconsistently',
    'inconvenience', 'inconveniences', 'inconvenient',
    'incorrect', 'incorrectly',
    'indecency', 'indecent',
    'indefeasible', 'indefeasibly',
    'indict', 'indicted', 'indicting', 'indictment', 'indictments', 'indicts',
    'ineffective', 'ineffectively', 'ineffectiveness',
    'inefficiencies', 'inefficiency', 'inefficient', 'inefficiently',
    'ineligibility', 'ineligible',
    'inequitable', 'inequitably', 'inequities', 'inequity',
    'inexperience', 'inexperienced',
    'inferior',
    'inflict', 'inflicted', 'inflicting', 'infliction', 'inflicts',
    'infraction', 'infractions',
    'infringe', 'infringed', 'infringement', 'infringements', 'infringes', 'infringing',
    'inhibit', 'inhibited', 'inhibiting', 'inhibits',
    'injunction', 'injunctions', 'injunctive',
    'injure', 'injured', 'injures', 'injuries', 'injuring', 'injurious', 'injury',
    'insolvent', 'insolvencies', 'insolvency',
    'instability', 'instabilities',
    'insufficient', 'insufficiently',
    'interfere', 'interfered', 'interference', 'interferences', 'interferes', 'interfering',
    'intermittent', 'intermittently',
    'interrupt', 'interrupted', 'interrupting', 'interruption', 'interruptions', 'interrupts',
    'intimidate', 'intimidated', 'intimidates', 'intimidating', 'intimidation',
    'invalid', 'invalidate', 'invalidated', 'invalidates', 'invalidating', 'invalidation', 'invalidity',
    'investigate', 'investigated', 'investigates', 'investigating', 'investigation', 'investigations', 'investigative', 'investigators',
    'involuntarily', 'involuntary',
    'irregularities', 'irregularity',
    'irreparable', 'irreparably',
    'jeopardize', 'jeopardized', 'jeopardizes', 'jeopardizing', 'jeopardy',
    'kill', 'killed', 'killing', 'kills',
    'lack', 'lacked', 'lacking', 'lacks',
    'lag', 'lagged', 'lagging', 'lags',
    'lapse', 'lapsed', 'lapses', 'lapsing',
    'late', 'later', 'latest', 'lateness',
    'lawsuit', 'lawsuits',
    'layoff', 'layoffs',
    'leak', 'leaked', 'leaking', 'leaks',
    'liability', 'liabilities', 'liable',
    'limitation', 'limitations', 'limiting',
    'liquidate', 'liquidated', 'liquidates', 'liquidating', 'liquidation', 'liquidations', 'liquidator', 'liquidators',
    'litigate', 'litigated', 'litigates', 'litigating', 'litigation', 'litigations',
    'lose', 'loser', 'losers', 'loses', 'losing',
    'loss', 'losses',
    'malfeasance',
    'malfunction', 'malfunctioned', 'malfunctioning', 'malfunctions',
    'malice', 'malicious', 'maliciously',
    'malpractice',
    'manipulate', 'manipulated', 'manipulates', 'manipulating', 'manipulation', 'manipulations', 'manipulative',
    'markdown', 'markdowns',
    'misappropriate', 'misappropriated', 'misappropriates', 'misappropriating', 'misappropriation', 'misappropriations',
    'miscalculate', 'miscalculated', 'miscalculates', 'miscalculating', 'miscalculation', 'miscalculations',
    'misconduct',
    'mishandle', 'mishandled', 'mishandles', 'mishandling',
    'misinform', 'misinformation', 'misinformed', 'misinforming', 'misinforms',
    'misinterpret', 'misinterpretation', 'misinterpretations', 'misinterpreted', 'misinterpreting', 'misinterprets',
    'mislead', 'misleading', 'misleads', 'misled',
    'mismanage', 'mismanaged', 'mismanagement', 'mismanages', 'mismanaging',
    'mismatch', 'mismatched', 'mismatches',
    'misrepresent', 'misrepresentation', 'misrepresentations', 'misrepresented', 'misrepresenting', 'misrepresents',
    'miss', 'missed', 'misses', 'missing',
    'misstate', 'misstated', 'misstatement', 'misstatements', 'misstates', 'misstating',
    'mistake', 'mistaken', 'mistakenly', 'mistakes',
    'mistrust', 'mistrusted', 'mistrusting', 'mistrusts',
    'misunderstand', 'misunderstanding', 'misunderstandings', 'misunderstands', 'misunderstood',
    'misuse', 'misused', 'misuses', 'misusing',
    'monopolistic', 'monopoly', 'monopolies', 'monopolize', 'monopolized', 'monopolizes', 'monopolizing',
    'moratoria', 'moratorium',
    'mothball', 'mothballed', 'mothballing', 'mothballs',
    'negate', 'negated', 'negates', 'negating', 'negation',
    'negative', 'negatively', 'negatives',
    'neglect', 'neglected', 'neglectful', 'neglecting', 'neglects',
    'negligence', 'negligent', 'negligently',
    'nonattainment',
    'noncompliance', 'noncompliant',
    'nonpayment', 'nonpayments',
    'nonperformance', 'nonperforming',
    'nonproductive',
    'object', 'objected', 'objecting', 'objection', 'objectionable', 'objections', 'objects',
    'obscene', 'obscenity',
    'obsolescence', 'obsolete',
    'obstacle', 'obstacles',
    'obstruct', 'obstructed', 'obstructing', 'obstruction', 'obstructions', 'obstructive', 'obstructs',
    'offence', 'offences', 'offend', 'offended', 'offender', 'offenders', 'offending', 'offends', 'offense', 'offenses', 'offensive',
    'omission', 'omissions', 'omit', 'omits', 'omitted', 'omitting',
    'onerous',
    'oppose', 'opposed', 'opposes', 'opposing', 'opposition',
    'outage', 'outages',
    'overbuild', 'overbuilding', 'overbuilds', 'overbuilt',
    'overburden', 'overburdened', 'overburdening', 'overburdens',
    'overcome',
    'overdue',
    'overestimate', 'overestimated', 'overestimates', 'overestimating', 'overestimation',
    'overload', 'overloaded', 'overloading', 'overloads',
    'overlook', 'overlooked', 'overlooking', 'overlooks',
    'overpaid', 'overpay', 'overpaying', 'overpayment', 'overpayments', 'overpays',
    'overproduced', 'overproduces', 'overproducing', 'overproduction',
    'overrun', 'overruns',
    'overstate', 'overstated', 'overstatement', 'overstatements', 'overstates', 'overstating',
    'oversupplied', 'oversupplies', 'oversupply',
    'overturn', 'overturned', 'overturning', 'overturns',
    'overvalue', 'overvalued', 'overvalues', 'overvaluing',
    'panic', 'panicked', 'panicking', 'panics',
    'penal', 'penalize', 'penalized', 'penalizes', 'penalizing', 'penalties', 'penalty',
    'peril', 'perilous', 'perils',
    'perjury',
    'perpetrate', 'perpetrated', 'perpetrates', 'perpetrating', 'perpetration',
    'persist', 'persisted', 'persistence', 'persistent', 'persistently', 'persisting', 'persists',
    'pervasive',
    'pessimism', 'pessimistic',
    'petition', 'petitioned', 'petitioning', 'petitions',
    'picket', 'picketed', 'picketing', 'pickets',
    'plaintiff', 'plaintiffs',
    'plea', 'plead', 'pleaded', 'pleading', 'pleadings', 'pleads', 'pleas', 'pled',
    'poor', 'poorer', 'poorest', 'poorly',
    'pose', 'posed', 'poses', 'posing',
    'precarious', 'precariously',
    'precaution', 'precautionary', 'precautions',
    'precipitate', 'precipitated', 'precipitately', 'precipitates', 'precipitating', 'precipitous', 'precipitously',
    'preclude', 'precluded', 'precludes', 'precluding', 'preclusion', 'preclusions', 'preclusive',
    'prejudice', 'prejudiced', 'prejudices', 'prejudicial', 'prejudicing',
    'premature', 'prematurely',
    'pressure', 'pressured', 'pressures', 'pressuring',
    'prevent', 'preventable', 'prevented', 'preventing', 'prevention', 'prevents',
    'problem', 'problematic', 'problematical', 'problems',
    'prolong', 'prolongation', 'prolonged', 'prolonging', 'prolongs',
    'prosecute', 'prosecuted', 'prosecutes', 'prosecuting', 'prosecution', 'prosecutions', 'prosecutor', 'prosecutors',
    'protest', 'protested', 'protester', 'protesters', 'protesting', 'protestor', 'protestors', 'protests',
    'protracted',
    'provoke', 'provoked', 'provokes', 'provoking',
    'punish', 'punishable', 'punished', 'punishes', 'punishing', 'punishment', 'punishments', 'punitive',
    'question', 'questionable', 'questioned', 'questioning', 'questions',
    'quit', 'quits', 'quitting',
    'racketeer', 'racketeering',
    'raid', 'raided', 'raiding', 'raids',
    'reassess', 'reassessed', 'reassesses', 'reassessing', 'reassessment', 'reassessments',
    'recall', 'recalled', 'recalling', 'recalls',
    'recession', 'recessions',
    'reckless', 'recklessly', 'recklessness',
    'redact', 'redacted', 'redacting', 'redaction', 'redactions', 'redacts',
    'redress', 'redressed', 'redresses', 'redressing',
    'reduce', 'reduced', 'reduces', 'reducing', 'reduction', 'reductions',
    'refrain', 'refrained', 'refraining', 'refrains',
    'refuse', 'refused', 'refuses', 'refusing',
    'reject', 'rejected', 'rejecting', 'rejection', 'rejections', 'rejects',
    'relinquish', 'relinquished', 'relinquishes', 'relinquishing', 'relinquishment',
    'reluctance', 'reluctant', 'reluctantly',
    'renegotiate', 'renegotiated', 'renegotiates', 'renegotiating', 'renegotiation', 'renegotiations',
    'repeal', 'repealed', 'repealing', 'repeals',
    'repossess', 'repossessed', 'repossesses', 'repossessing', 'repossession', 'repossessions',
    'reprimand', 'reprimanded', 'reprimanding', 'reprimands',
    'repudiate', 'repudiated', 'repudiates', 'repudiating', 'repudiation', 'repudiations',
    'rescind', 'rescinded', 'rescinding', 'rescinds', 'rescission', 'rescissions',
    'resign', 'resignation', 'resignations', 'resigned', 'resigning', 'resigns',
    'restate', 'restated', 'restatement', 'restatements', 'restates', 'restating',
    'restitution', 'restitutions',
    'restrain', 'restrained', 'restraining', 'restrains', 'restraint', 'restraints',
    'restrict', 'restricted', 'restricting', 'restriction', 'restrictions', 'restrictive', 'restricts',
    'restructure', 'restructured', 'restructures', 'restructuring', 'restructurings',
    'retaliate', 'retaliated', 'retaliates', 'retaliating', 'retaliation', 'retaliations', 'retaliatory',
    'retract', 'retracted', 'retracting', 'retraction', 'retractions', 'retracts',
    'retrench', 'retrenched', 'retrenches', 'retrenching', 'retrenchment', 'retrenchments',
    'revocation', 'revocations', 'revoke', 'revoked', 'revokes', 'revoking',
    'ridicule', 'ridiculed', 'ridicules', 'ridiculing', 'ridiculous', 'ridiculously',
    'risk', 'risked', 'riskier', 'riskiest', 'risking', 'risks', 'risky',
    'sabotage', 'sabotaged', 'sabotages', 'sabotaging',
    'sacrifice', 'sacrificed', 'sacrifices', 'sacrificial', 'sacrificing',
    'sanction', 'sanctioned', 'sanctioning', 'sanctions',
    'scandal', 'scandals', 'scandalous',
    'scarcities', 'scarcity',
    'scrutinize', 'scrutinized', 'scrutinizes', 'scrutinizing', 'scrutiny',
    'seize', 'seized', 'seizes', 'seizing', 'seizure', 'seizures',
    'serious', 'seriously', 'seriousness',
    'setback', 'setbacks',
    'sever', 'severance', 'severances', 'severe', 'severed', 'severely', 'severer', 'severest', 'severing', 'severity', 'severs',
    'sharply',
    'shock', 'shocked', 'shocking', 'shocks',
    'shortage', 'shortages',
    'shortcoming', 'shortcomings',
    'shortfall', 'shortfalls',
    'shrink', 'shrinkage', 'shrinking', 'shrinks', 'shrunk',
    'shut', 'shutdown', 'shutdowns', 'shuts', 'shutting',
    'slander', 'slandered', 'slandering', 'slanderous', 'slanders',
    'slash', 'slashed', 'slashes', 'slashing',
    'slip', 'slippage', 'slipped', 'slipping', 'slips',
    'slow', 'slowdown', 'slowdowns', 'slowed', 'slower', 'slowest', 'slowing', 'slowly', 'slows',
    'slump', 'slumped', 'slumping', 'slumps',
    'smuggle', 'smuggled', 'smuggler', 'smugglers', 'smuggles', 'smuggling',
    'squeeze', 'squeezed', 'squeezes', 'squeezing',
    'stagnant', 'stagnate', 'stagnated', 'stagnates', 'stagnating', 'stagnation',
    'stall', 'stalled', 'stalling', 'stalls',
    'standstill', 'standstills',
    'stolen',
    'stop', 'stoppage', 'stoppages', 'stopped', 'stopping', 'stops',
    'strain', 'strained', 'straining', 'strains',
    'stress', 'stressed', 'stresses', 'stressful', 'stressing',
    'strike', 'strikebreaker', 'strikebreakers', 'strikes', 'striking',
    'stringent', 'stringently',
    'struggle', 'struggled', 'struggles', 'struggling',
    'subpoena', 'subpoenaed', 'subpoenas',
    'substandard',
    'sue', 'sued', 'sues', 'suing',
    'suffer', 'suffered', 'suffering', 'suffers',
    'summons', 'summonsed', 'summonses',
    'superfund',
    'surplus', 'surpluses',
    'susceptibility', 'susceptible',
    'suspect', 'suspected', 'suspecting', 'suspects',
    'suspend', 'suspended', 'suspending', 'suspends', 'suspension', 'suspensions',
    'suspicion', 'suspicions', 'suspicious', 'suspiciously',
    'taint', 'tainted', 'tainting', 'taints',
    'tamper', 'tampered', 'tampering', 'tampers',
    'tardy',
    'terminate', 'terminated', 'terminates', 'terminating', 'termination', 'terminations',
    'terrible', 'terribly',
    'theft', 'thefts',
    'threat', 'threaten', 'threatened', 'threatening', 'threatens', 'threats',
    'tolerate', 'tolerated', 'tolerates', 'tolerating',
    'torture', 'tortured', 'tortures', 'torturing',
    'toxic', 'toxicities', 'toxicity', 'toxics',
    'trauma', 'traumas', 'traumatic',
    'trouble', 'troubled', 'troubles', 'troubling',
    'unable',
    'unacceptable', 'unacceptably',
    'unaccounted',
    'unanticipated',
    'unapproved',
    'unattractive',
    'unauthorized',
    'unavailability', 'unavailable',
    'unavoidable', 'unavoidably',
    'unaware',
    'uncertain', 'uncertainly', 'uncertainties', 'uncertainty',
    'uncollectable', 'uncollected', 'uncollectible',
    'uncompetitive',
    'unconscionable', 'unconscionably',
    'unconstitutional', 'unconstitutionality', 'unconstitutionally',
    'uncontrollable', 'uncontrollably', 'uncontrolled',
    'uncorrected',
    'uncover', 'uncovered', 'uncovering', 'uncovers',
    'undercut', 'undercuts', 'undercutting',
    'underestimate', 'underestimated', 'underestimates', 'underestimating', 'underestimation',
    'underfunded', 'underfunding',
    'underinsured',
    'undermine', 'undermined', 'undermines', 'undermining',
    'underpaid', 'underpay', 'underpaying', 'underpayment', 'underpayments', 'underpays',
    'underperform', 'underperformance', 'underperformed', 'underperforming', 'underperforms',
    'underreport', 'underreported', 'underreporting', 'underreports',
    'understate', 'understated', 'understatement', 'understatements', 'understates', 'understating',
    'undervalue', 'undervalued', 'undervalues', 'undervaluing',
    'underweight',
    'undesirable', 'undesirably',
    'undetected',
    'undisclosed',
    'undocumented',
    'undue', 'unduly',
    'uneconomic', 'uneconomical', 'uneconomically',
    'unemploy', 'unemployed', 'unemployment',
    'unenforceable', 'unenforceability',
    'unethical', 'unethically',
    'unexpected', 'unexpectedly',
    'unfair', 'unfairly', 'unfairness',
    'unfavorable', 'unfavorably', 'unfavourable', 'unfavourably',
    'unfit',
    'unforeseen',
    'unfortunate', 'unfortunately',
    'unfounded',
    'unfriendly',
    'unfulfilled',
    'unfunded',
    'uninsured',
    'unintended', 'unintentional', 'unintentionally',
    'unjust', 'unjustified', 'unjustly',
    'unknown', 'unknowingly', 'unknowns',
    'unlawful', 'unlawfully',
    'unlicensed',
    'unlikely',
    'unliquidated',
    'unmarketable',
    'unmerchantable', 'unmerchantability',
    'unmet',
    'unnecessary', 'unnecessarily',
    'unneeded',
    'unobtainable',
    'unoccupied',
    'unpaid',
    'unperformed',
    'unplanned',
    'unpopular',
    'unprecedented', 'unprecedentedly',
    'unpredictability', 'unpredictable', 'unpredictably',
    'unprepared',
    'unproductive',
    'unprofitability', 'unprofitable',
    'unqualified',
    'unreasonable', 'unreasonableness', 'unreasonably',
    'unrecorded',
    'unrecoverable', 'unrecovered',
    'unreimbursed',
    'unreliable', 'unreliability',
    'unremediable', 'unremedied',
    'unresolved',
    'unrest',
    'unsafe',
    'unsalable', 'unsaleable',
    'unsatisfactory', 'unsatisfactorily', 'unsatisfied',
    'unscheduled',
    'unsold',
    'unsound',
    'unstabilized', 'unstable',
    'unsubstantiated',
    'unsuccessful', 'unsuccessfully',
    'unsuitability', 'unsuitable', 'unsuitably', 'unsuited',
    'unsupported',
    'unsure',
    'unsuspecting',
    'untenable',
    'untimely',
    'untrue',
    'unusual', 'unusually',
    'unwanted',
    'unwarranted',
    'unwilling', 'unwillingness',
    'upend', 'upended', 'upending', 'upends',
    'upset', 'upsets', 'upsetting',
    'urgent', 'urgently',
    'usurp', 'usurped', 'usurping', 'usurps',
    'vandalism', 'vandalize', 'vandalized', 'vandalizes', 'vandalizing',
    'verdict', 'verdicts',
    'veto', 'vetoed', 'vetoes', 'vetoing',
    'violate', 'violated', 'violates', 'violating', 'violation', 'violations', 'violator', 'violators',
    'violence', 'violent', 'violently',
    'volatile', 'volatilities', 'volatility',
    'vulnerable', 'vulnerabilities', 'vulnerability',
    'warn', 'warned', 'warning', 'warnings', 'warns',
    'wasted', 'wasteful', 'wasting',
    'weak', 'weaken', 'weakened', 'weakening', 'weakens', 'weaker', 'weakest', 'weakly', 'weakness', 'weaknesses',
    'willfully',
    'withdraw', 'withdrawal', 'withdrawals', 'withdrawing', 'withdrawn', 'withdraws', 'withdrew',
    'withhold', 'withheld', 'withholding', 'withholds',
    'worry', 'worries', 'worrisome', 'worrying',
    'worse', 'worsen', 'worsened', 'worsening', 'worsens', 'worst',
    'worthless', 'worthlessness',
    'writedown', 'writedowns', 'writeoff', 'writeoffs',
    'wrong', 'wrongdoing', 'wrongdoings', 'wrongful', 'wrongfully', 'wrongly',
}

# Loughran-McDonald Positive Words
LM_POSITIVE = {
    'able', 'abundance', 'abundant', 'acclaimed', 'accomplish', 'accomplished', 'accomplishes',
    'accomplishing', 'accomplishment', 'accomplishments', 'achieve', 'achieved', 'achievement',
    'achievements', 'achieves', 'achieving', 'adequate', 'admirably', 'advance', 'advanced',
    'advancement', 'advancements', 'advances', 'advancing', 'advantage', 'advantaged',
    'advantageous', 'advantageously', 'advantages', 'alliance', 'alliances', 'assure',
    'assured', 'assures', 'assuring', 'attain', 'attained', 'attaining', 'attainment',
    'attainments', 'attains', 'attractive', 'attractively', 'attractiveness', 'beautiful',
    'beautifully', 'beneficial', 'beneficially', 'benefit', 'benefited', 'benefiting',
    'benefits', 'benefitted', 'benefitting', 'best', 'better', 'bolster', 'bolstered',
    'bolstering', 'bolsters', 'boom', 'booming', 'boost', 'boosted', 'boosting', 'boosts',
    'breakthrough', 'breakthroughs', 'bright', 'brighten', 'brightened', 'brightening',
    'brighter', 'brightest', 'brilliance', 'brilliant', 'brilliantly', 'charitable',
    'collaborate', 'collaborated', 'collaborates', 'collaborating', 'collaboration',
    'collaborations', 'collaborative', 'collaborator', 'collaborators', 'commend',
    'commendable', 'commendably', 'commended', 'commending', 'commends', 'compliment',
    'complimentary', 'complimented', 'complimenting', 'compliments', 'conclusive',
    'conclusively', 'conducive', 'confident', 'confidently', 'constructive', 'constructively',
    'courteous', 'creative', 'creatively', 'creativeness', 'creativity', 'delight',
    'delighted', 'delightful', 'delightfully', 'delighting', 'delights', 'dependability',
    'dependable', 'dependably', 'desirable', 'desirably', 'despite', 'destined',
    'diligence', 'diligent', 'diligently', 'distinction', 'distinctions', 'distinctive',
    'distinctively', 'distinctiveness', 'dream', 'dreams', 'easier', 'easiest', 'easily',
    'easy', 'effective', 'efficiencies', 'efficiency', 'efficient', 'efficiently',
    'empower', 'empowered', 'empowering', 'empowerment', 'empowers', 'enable', 'enabled',
    'enables', 'enabling', 'encourage', 'encouraged', 'encouragement', 'encourages',
    'encouraging', 'enhance', 'enhanced', 'enhancement', 'enhancements', 'enhances',
    'enhancing', 'enjoy', 'enjoyable', 'enjoyably', 'enjoyed', 'enjoying', 'enjoyment',
    'enjoys', 'enthusiasm', 'enthusiastic', 'enthusiastically', 'entrepreneurial',
    'excellence', 'excellent', 'exceptionally', 'excited', 'excitement', 'exciting',
    'exclusive', 'exclusively', 'exclusiveness', 'exclusives', 'exclusivity', 'exemplary',
    'fabulous', 'fabulously', 'fair', 'fairly', 'famous', 'famously', 'fantastic',
    'fantastically', 'favorable', 'favorably', 'favored', 'favoring', 'favorite',
    'favorites', 'favour', 'favourable', 'favourably', 'favoured', 'favouring', 'favourite',
    'favourites', 'finest', 'first-class', 'flourish', 'flourished', 'flourishes',
    'flourishing', 'foremost', 'fortunate', 'fortunately', 'fortune', 'fortunes', 'free',
    'freed', 'freedom', 'freedoms', 'freeing', 'freely', 'friendly', 'gain', 'gained',
    'gaining', 'gains', 'generous', 'generously', 'good', 'goodness', 'goodwill', 'grace',
    'graceful', 'gracefully', 'graces', 'gracious', 'graciously', 'graciousness', 'grand',
    'grandest', 'grandly', 'great', 'greater', 'greatest', 'greatly', 'greatness', 'grew',
    'grow', 'growing', 'grown', 'grows', 'growth', 'guidance', 'guide', 'guided', 'guides',
    'guiding', 'happiness', 'happy', 'hardest', 'healthy', 'helpful', 'helpfully',
    'helpfulness', 'highest', 'honor', 'honorable', 'honorably', 'honored', 'honoring',
    'honors', 'honour', 'honourable', 'honourably', 'honoured', 'honouring', 'honours',
    'hope', 'hoped', 'hopeful', 'hopefully', 'hopefulness', 'hopes', 'hoping', 'ideal',
    'ideally', 'immense', 'immensely', 'impress', 'impressed', 'impresses', 'impressing',
    'impressive', 'impressively', 'improve', 'improved', 'improvement', 'improvements',
    'improves', 'improving', 'incredible', 'incredibly', 'influential', 'informative',
    'ingenuity', 'innovate', 'innovated', 'innovates', 'innovating', 'innovation',
    'innovations', 'innovative', 'innovativeness', 'innovator', 'innovators', 'insightful',
    'insightfully', 'inspiration', 'inspirational', 'inspire', 'inspired', 'inspires',
    'inspiring', 'integrity', 'invent', 'invented', 'inventing', 'invention', 'inventions',
    'inventive', 'inventiveness', 'inventor', 'inventors', 'invents', 'leadership',
    'leading', 'loyal', 'loyalty', 'lucrative', 'marvelous', 'marvelously', 'marvel',
    'marvels', 'maximize', 'maximized', 'maximizes', 'maximizing', 'meritorious',
    'meritoriously', 'milestone', 'milestones', 'nice', 'nicely', 'nicest', 'noble',
    'noblest', 'nobly', 'notable', 'notably', 'noteworthy', 'opportunities', 'opportunity',
    'optimal', 'optimism', 'optimistic', 'optimistically', 'optimize', 'optimized',
    'optimizes', 'optimizing', 'optimum', 'outpace', 'outpaced', 'outpaces', 'outpacing',
    'outperform', 'outperformed', 'outperforming', 'outperforms', 'outstanding',
    'outstandingly', 'overcome', 'overcomes', 'overcoming', 'perfect', 'perfected',
    'perfecting', 'perfection', 'perfectly', 'pleasant', 'pleasantly', 'please', 'pleased',
    'pleases', 'pleasing', 'pleasingly', 'pleasure', 'pleasurable', 'pleasurably',
    'pleasures', 'plentiful', 'plentifully', 'plenty', 'popular', 'popularity', 'popularly',
    'positive', 'positively', 'positives', 'powerful', 'powerfully', 'praise', 'praised',
    'praises', 'praising', 'preeminent', 'preeminently', 'premier', 'premiere', 'prestige',
    'prestigious', 'pride', 'proactive', 'proactively', 'proficiency', 'proficient',
    'proficiently', 'profit', 'profitability', 'profitable', 'profitably', 'profited',
    'profiting', 'profits', 'progress', 'progressed', 'progresses', 'progressing',
    'progressive', 'progressively', 'prominent', 'prominently', 'promise', 'promised',
    'promises', 'promising', 'prosper', 'prospered', 'prospering', 'prosperity',
    'prosperous', 'prosperously', 'prospers', 'proud', 'proudly', 'prove', 'proved',
    'proven', 'proves', 'proving', 'prudence', 'prudent', 'prudently', 'quality',
    'reassure', 'reassured', 'reassures', 'reassuring', 'reassuringly', 'recognition',
    'recognize', 'recognized', 'recognizes', 'recognizing', 'recommend', 'recommendation',
    'recommendations', 'recommended', 'recommending', 'recommends', 'record', 'records',
    'recover', 'recovered', 'recovering', 'recovers', 'recovery', 'refund', 'refunded',
    'refunding', 'refunds', 'regain', 'regained', 'regaining', 'regains', 'reinforce',
    'reinforced', 'reinforcement', 'reinforcements', 'reinforces', 'reinforcing',
    'rejoice', 'rejoiced', 'rejoices', 'rejoicing', 'reliability', 'reliable', 'reliably',
    'relieve', 'relieved', 'relieves', 'relieving', 'remarkable', 'remarkably', 'renown',
    'renowned', 'reputable', 'reputably', 'reputation', 'reputations', 'resolve', 'resolved',
    'resolves', 'resolving', 'respect', 'respected', 'respecting', 'respects', 'restore',
    'restored', 'restores', 'restoring', 'revitalize', 'revitalized', 'revitalizes',
    'revitalizing', 'reward', 'rewarded', 'rewarding', 'rewards', 'rich', 'richer',
    'richest', 'richly', 'robust', 'robustly', 'robustness', 'safe', 'safely', 'safeness',
    'safer', 'safest', 'safety', 'satisfaction', 'satisfactorily', 'satisfactory',
    'satisfied', 'satisfies', 'satisfy', 'satisfying', 'save', 'saved', 'saves', 'saving',
    'savings', 'seamless', 'seamlessly', 'secure', 'secured', 'securely', 'secures',
    'securing', 'security', 'simple', 'simpler', 'simplest', 'simplified', 'simplifies',
    'simplify', 'simplifying', 'simplistic', 'simply', 'smooth', 'smoothed', 'smoother',
    'smoothest', 'smoothing', 'smoothly', 'smooths', 'solid', 'solidified', 'solidifies',
    'solidify', 'solidifying', 'solidly', 'solids', 'solution', 'solutions', 'solve',
    'solved', 'solves', 'solving', 'spectacular', 'spectacularly', 'stability', 'stabilization',
    'stabilize', 'stabilized', 'stabilizes', 'stabilizing', 'stable', 'stably', 'star',
    'stars', 'stellar', 'straightforward', 'straightforwardly', 'strength', 'strengthen',
    'strengthened', 'strengthening', 'strengthens', 'strengths', 'strong', 'stronger',
    'strongest', 'strongly', 'succeed', 'succeeded', 'succeeding', 'succeeds', 'success',
    'successes', 'successful', 'successfully', 'superior', 'superiorly', 'superiority',
    'supremacy', 'surpass', 'surpassed', 'surpasses', 'surpassing', 'terrific',
    'terrifically', 'thrill', 'thrilled', 'thrilling', 'thrillingly', 'thrills', 'thrive',
    'thrived', 'thrives', 'thriving', 'top', 'tops', 'transparent', 'transparently',
    'tremendous', 'tremendously', 'triumph', 'triumphal', 'triumphant', 'triumphantly',
    'triumphed', 'triumphing', 'triumphs', 'trust', 'trusted', 'trusting', 'trusts',
    'trustworthiness', 'trustworthy', 'truth', 'truthful', 'truthfully', 'truthfulness',
    'unbeatable', 'unmatched', 'unprecedented', 'unsurpassed', 'upturn', 'upturns',
    'upward', 'upwardly', 'upwards', 'valuable', 'valuably', 'value', 'valued', 'values',
    'versatile', 'versatility', 'viability', 'viable', 'viably', 'vibrant', 'vibrantly',
    'victory', 'victories', 'vigorous', 'vigorously', 'virtue', 'virtues', 'virtuous',
    'virtuously', 'visionary', 'win', 'winner', 'winners', 'winning', 'wins', 'won',
    'wonderful', 'wonderfully', 'worthiness', 'worthwhile', 'worthy',
}

# Loughran-McDonald Uncertainty Words
LM_UNCERTAINTY = {
    'abeyance', 'abeyances', 'almost', 'ambiguity', 'ambiguous', 'anomalies', 'anomalous',
    'anomaly', 'anticipate', 'anticipated', 'anticipates', 'anticipating', 'apparently',
    'appear', 'appeared', 'appearing', 'appears', 'approximate', 'approximated',
    'approximately', 'approximates', 'approximating', 'approximation', 'approximations',
    'arbitrarily', 'arbitrary', 'assume', 'assumed', 'assumes', 'assuming', 'assumption',
    'assumptions', 'believe', 'believed', 'believes', 'believing', 'cautious', 'cautiously',
    'clarification', 'clarifications', 'confuse', 'confused', 'confuses', 'confusing',
    'confusingly', 'confusion', 'contingencies', 'contingency', 'contingent',
    'contingently', 'contingents', 'could', 'crossroad', 'crossroads', 'depend',
    'depended', 'dependence', 'dependencies', 'dependency', 'dependent', 'depending',
    'depends', 'destabilize', 'destabilized', 'destabilizes', 'destabilizing', 'deviate',
    'deviated', 'deviates', 'deviating', 'deviation', 'deviations', 'differ', 'differed',
    'differing', 'differs', 'doubt', 'doubted', 'doubtful', 'doubting', 'doubts',
    'exposing', 'exposure', 'exposures', 'fluctuate', 'fluctuated', 'fluctuates',
    'fluctuating', 'fluctuation', 'fluctuations', 'hidden', 'imprecise', 'imprecision',
    'imprecisions', 'improbability', 'improbable', 'incompleteness', 'indefinite',
    'indefinitely', 'indefiniteness', 'indeterminable', 'indeterminate', 'instabilities',
    'instability', 'intangible', 'intangibles', 'likelihood', 'may', 'maybe', 'might',
    'nearly', 'occasionally', 'pending', 'perhaps', 'possibilities', 'possibility',
    'possible', 'possibly', 'precaution', 'precautionary', 'precautions', 'predict',
    'predictability', 'predicted', 'predicting', 'prediction', 'predictions', 'predicts',
    'preliminarily', 'preliminary', 'presumably', 'presume', 'presumed', 'presumes',
    'presuming', 'presumption', 'presumptions', 'probabilistic', 'probabilities',
    'probability', 'probable', 'probably', 'random', 'randomly', 'randomness', 'reassess',
    'reassessed', 'reassesses', 'reassessing', 'reassessment', 'reassessments',
    'recalculate', 'recalculated', 'recalculates', 'recalculating', 'recalculation',
    'reconsider', 'reconsidered', 'reconsidering', 'reconsiders', 'reexamine',
    'reexamined', 'reexamines', 'reexamining', 'reinterpret', 'reinterpretation',
    'reinterpreted', 'reinterpreting', 'reinterprets', 'revise', 'revised', 'revises',
    'revising', 'revision', 'revisions', 'risk', 'risked', 'riskier', 'riskiest',
    'risking', 'risks', 'risky', 'roughly', 'rumors', 'seem', 'seemed', 'seeming',
    'seemingly', 'seems', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'speculate',
    'speculated', 'speculates', 'speculating', 'speculation', 'speculations', 'speculative',
    'sudden', 'suddenly', 'suggest', 'suggested', 'suggesting', 'suggestion', 'suggestions',
    'suggests', 'susceptibility', 'susceptible', 'tending', 'tentative', 'tentatively',
    'turbulence', 'uncertain', 'uncertainly', 'uncertainties', 'uncertainty', 'unclear',
    'unconfirmed', 'undecided', 'undefined', 'undesignated', 'undetectable', 'undeterminable',
    'undetermined', 'unestablished', 'unexpected', 'unexpectedly', 'unforeseen', 'unidentified',
    'unknown', 'unknowns', 'unobservable', 'unpredictable', 'unpredictably', 'unpredicted',
    'unproven', 'unquantifiable', 'unquantified', 'unreliable', 'unresolved', 'unsettled',
    'unspecified', 'untested', 'unusual', 'unusually', 'unvested', 'unwritten', 'vagaries',
    'vague', 'vaguely', 'vagueness', 'variability', 'variable', 'variables', 'variably',
    'variance', 'variances', 'variant', 'variants', 'variation', 'variations', 'varied',
    'varies', 'vary', 'varying', 'volatile', 'volatilities', 'volatility',
}

# Loughran-McDonald Litigious Words (subset)
LM_LITIGIOUS = {
    'adjudicate', 'adjudicated', 'adjudicates', 'adjudicating', 'adjudication',
    'allegation', 'allegations', 'allege', 'alleged', 'allegedly', 'alleges', 'alleging',
    'antitrust', 'appeal', 'appealed', 'appealing', 'appeals', 'appellant', 'appellants',
    'appellate', 'arbitrate', 'arbitrated', 'arbitrates', 'arbitrating', 'arbitration',
    'arbitrations', 'arbitrator', 'arbitrators', 'claimant', 'claimants', 'class-action',
    'complaint', 'complaints', 'convict', 'convicted', 'convicting', 'conviction',
    'convictions', 'convicts', 'counterclaim', 'counterclaimed', 'counterclaiming',
    'counterclaims', 'court', 'courts', 'crime', 'crimes', 'criminal', 'criminally',
    'criminals', 'damages', 'decree', 'decreed', 'decrees', 'decreeing', 'defendant',
    'defendants', 'deposition', 'depositions', 'discovery', 'discriminate', 'discriminated',
    'discriminates', 'discriminating', 'discrimination', 'discriminatory', 'dispute',
    'disputed', 'disputes', 'disputing', 'felonies', 'felony', 'fraud', 'frauds',
    'fraudulent', 'fraudulently', 'guilt', 'guilty', 'indict', 'indicted', 'indictment',
    'indictments', 'indicts', 'infraction', 'infractions', 'infringe', 'infringed',
    'infringement', 'infringements', 'infringes', 'infringing', 'injunction', 'injunctions',
    'injunctive', 'jury', 'law', 'lawful', 'lawfully', 'laws', 'lawsuit', 'lawsuits',
    'lawyer', 'lawyers', 'legal', 'legality', 'legally', 'legislation', 'legislations',
    'legislative', 'libel', 'libelous', 'libels', 'litigate', 'litigated', 'litigates',
    'litigating', 'litigation', 'litigations', 'mislead', 'misleading', 'misleads',
    'misled', 'misrepresent', 'misrepresentation', 'misrepresentations', 'misrepresented',
    'misrepresenting', 'misrepresents', 'motion', 'motioned', 'motioning', 'motions',
    'negligence', 'negligent', 'negligently', 'perjury', 'petition', 'petitioned',
    'petitioning', 'petitions', 'plaintiff', 'plaintiffs', 'plea', 'plead', 'pleaded',
    'pleading', 'pleadings', 'pleads', 'pleas', 'pled', 'prosecute', 'prosecuted',
    'prosecutes', 'prosecuting', 'prosecution', 'prosecutions', 'prosecutor', 'prosecutors',
    'regulatory', 'respondent', 'respondents', 'restitution', 'sanction', 'sanctioned',
    'sanctioning', 'sanctions', 'settlement', 'settlements', 'settling', 'slander',
    'slanderous', 'slanders', 'statute', 'statutes', 'statutory', 'subpoena', 'subpoenaed',
    'subpoenas', 'sue', 'sued', 'sues', 'suing', 'suit', 'suits', 'summons', 'testimony',
    'tribunal', 'tribunals', 'verdict', 'verdicts', 'violate', 'violated', 'violates',
    'violating', 'violation', 'violations', 'violator', 'violators', 'witness', 'witnesses',
}

# Constraining words (indicate financial constraints)
LM_CONSTRAINING = {
    'abide', 'abides', 'abiding', 'bound', 'bounded', 'commit', 'commitment', 'commitments',
    'commits', 'committed', 'committing', 'compel', 'compelled', 'compelling', 'compels',
    'comply', 'compulsion', 'compulsory', 'condition', 'conditioned', 'conditions',
    'confine', 'confined', 'confines', 'confining', 'constrain', 'constrained', 'constraining',
    'constrains', 'constraint', 'constraints', 'decree', 'decreed', 'decrees', 'decreeing',
    'dictate', 'dictated', 'dictates', 'dictating', 'encumber', 'encumbered', 'encumbering',
    'encumbers', 'encumbrance', 'encumbrances', 'forbid', 'forbidden', 'forbidding',
    'forbids', 'force', 'forced', 'forces', 'forcing', 'impair', 'impaired', 'impairing',
    'impairment', 'impairments', 'impairs', 'impede', 'impeded', 'impedes', 'impediment',
    'impediments', 'impeding', 'impose', 'imposed', 'imposes', 'imposing', 'imposition',
    'impositions', 'inhibit', 'inhibited', 'inhibiting', 'inhibits', 'limit', 'limitation',
    'limitations', 'limited', 'limiting', 'limits', 'mandate', 'mandated', 'mandates',
    'mandating', 'mandatorily', 'mandatory', 'must', 'necessitate', 'necessitated',
    'necessitates', 'necessitating', 'noncompliance', 'noncompliant', 'oblige', 'obliged',
    'obliges', 'obliging', 'obligation', 'obligations', 'obligatory', 'preclude',
    'precluded', 'precludes', 'precluding', 'preclusion', 'preclusions', 'preclusive',
    'prescribe', 'prescribed', 'prescribes', 'prescribing', 'prescription', 'prescriptions',
    'prohibit', 'prohibited', 'prohibiting', 'prohibition', 'prohibitions', 'prohibitive',
    'prohibitively', 'prohibits', 'require', 'required', 'requirement', 'requirements',
    'requires', 'requiring', 'requisite', 'requisites', 'restrict', 'restricted',
    'restricting', 'restriction', 'restrictions', 'restrictive', 'restricts', 'restrain',
    'restrained', 'restraining', 'restrains', 'restraint', 'restraints', 'shall',
    'stringent', 'stringently',
}


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TONE_MODEL_CONFIG = {
    # Signal thresholds for net tone change
    'bullish_threshold': 0.03,      # Net tone improvement > 3%
    'bearish_threshold': -0.03,     # Net tone decline > 3%
    
    # Strength thresholds
    'strong_change': 0.06,          # Strong signal if change > 6%
    
    # Expected alpha (from academic research)
    'bullish_alpha': 0.03,          # +3% expected return for bullish
    'bearish_alpha': -0.025,        # -2.5% expected return for bearish
    
    # Holding period (days)
    'holding_period': 90,           # ~3 months
    
    # Signal decay
    'signal_decay_days': 120,       # Signal loses strength after 4 months
    
    # Minimum word count for reliable analysis
    'min_words': 1000,
    
    # SEC API settings
    'max_retries': 3,
    'request_delay': 0.2,           # SEC rate limit: 10 req/sec
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ToneMetrics:
    """Tone metrics for a single document."""
    total_words: int
    
    # Raw word counts
    negative_count: int = 0
    positive_count: int = 0
    uncertainty_count: int = 0
    litigious_count: int = 0
    constraining_count: int = 0
    
    # Proportions (normalized by total words)
    negative_pct: float = 0.0
    positive_pct: float = 0.0
    uncertainty_pct: float = 0.0
    litigious_pct: float = 0.0
    constraining_pct: float = 0.0
    
    # Net tone = (positive - negative) / (positive + negative)
    net_tone: float = 0.0
    
    # Filing metadata
    filing_date: Optional[datetime] = None
    filing_type: str = ""
    accession_number: str = ""


@dataclass
class ToneChangeSignal:
    """Signal from year-over-year tone change analysis."""
    ticker: str
    
    # Current vs prior year metrics
    current_tone: Optional[ToneMetrics] = None
    prior_tone: Optional[ToneMetrics] = None
    
    # Changes (current - prior)
    net_tone_change: float = 0.0
    negative_change: float = 0.0
    positive_change: float = 0.0
    uncertainty_change: float = 0.0
    litigious_change: float = 0.0
    constraining_change: float = 0.0
    
    # Signal
    direction: Optional[Literal["bullish", "bearish"]] = None
    strength: Literal["strong", "moderate", "weak", "none"] = "none"
    is_actionable: bool = False
    
    # Expected outcomes
    expected_alpha: Optional[float] = None
    holding_period_days: int = 90
    
    # Signal freshness
    days_since_filing: int = 0
    signal_decay: float = 1.0  # 1.0 = fresh, 0.0 = expired
    
    # Summary
    summary: str = ""
    rationale: str = ""
    concerns: List[str] = field(default_factory=list)


# ==============================================================================
# FILING TONE MODEL
# ==============================================================================

class FilingToneModel:
    """
    Model that analyzes 10-K tone changes to predict stock returns.
    
    Based on Loughran & McDonald (2011) financial sentiment dictionaries
    and academic research showing tone changes predict returns.
    """
    
    SEC_BASE_URL = "https://data.sec.gov"
    SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    
    # CIK lookup cache (same as SEC Filing Agent)
    CIK_CACHE = {
        "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
        "GOOG": "0001652044", "AMZN": "0001018724", "META": "0001326801",
        "NVDA": "0001045810", "TSLA": "0001318605", "JPM": "0000019617",
        "JNJ": "0000200406", "V": "0001403161", "UNH": "0000731766",
        "XOM": "0000034088", "PG": "0000080424", "MA": "0001141391",
        "HD": "0000354950", "CVX": "0000093410", "MRK": "0000310158",
        "ABBV": "0001551152", "PFE": "0000078003", "KO": "0000021344",
        "PEP": "0000077476", "COST": "0000909832", "AVGO": "0001730168",
        "WMT": "0000104169", "CSCO": "0000858877", "CRM": "0001108524",
        "AMD": "0000002488", "INTC": "0000050863", "ORCL": "0001341439",
        "NFLX": "0001065280", "ADBE": "0000796343", "LLY": "0000059478",
        "BAC": "0000070858", "GS": "0000886982", "MS": "0000895421",
        "C": "0000831001", "WFC": "0000072971", "BLK": "0001364742",
    }
    
    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize the filing tone model.
        
        Args:
            user_agent: SEC API user agent (name + email). If not provided,
                       will look for SEC_API_USER_AGENT env variable.
        """
        self.user_agent = user_agent or os.environ.get(
            "SEC_API_USER_AGENT", 
            "AutoResearcher research@example.com"
        )
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        
        # Pre-compile word sets for faster lookup
        self._negative_words = LM_NEGATIVE
        self._positive_words = LM_POSITIVE
        self._uncertainty_words = LM_UNCERTAINTY
        self._litigious_words = LM_LITIGIOUS
        self._constraining_words = LM_CONSTRAINING
        
        # Cache for tone metrics
        self._tone_cache: Dict[str, ToneMetrics] = {}
        
        logger.info("Initialized FilingToneModel with Loughran-McDonald dictionaries")
    
    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        ticker = ticker.upper()
        
        if ticker in self.CIK_CACHE:
            return self.CIK_CACHE[ticker]
        
        # Try to look up via SEC EDGAR
        try:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&output=atom"
            response = self.session.get(url, timeout=10)
            
            if response.ok:
                # Parse CIK from response
                match = re.search(r'CIK=(\d{10})', response.text)
                if match:
                    cik = match.group(1)
                    self.CIK_CACHE[ticker] = cik
                    return cik
        except Exception as e:
            logger.warning(f"Failed to look up CIK for {ticker}: {e}")
        
        return None
    
    def _get_10k_filings(self, ticker: str, num_filings: int = 2) -> List[Dict]:
        """
        Get recent 10-K filing metadata for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            num_filings: Number of recent 10-K filings to retrieve.
            
        Returns:
            List of filing metadata dicts with accession_number, filing_date, etc.
        """
        cik = self._get_cik(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for {ticker}")
            return []
        
        try:
            # Get company submissions
            url = self.SEC_SUBMISSIONS_URL.format(cik=cik)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            filings_data = data.get("filings", {}).get("recent", {})
            
            forms = filings_data.get("form", [])
            dates = filings_data.get("filingDate", [])
            accession_nums = filings_data.get("accessionNumber", [])
            primary_docs = filings_data.get("primaryDocument", [])
            
            filings = []
            for i, form in enumerate(forms):
                if form in ("10-K", "10-K/A") and len(filings) < num_filings:
                    accession = accession_nums[i].replace("-", "")
                    # CIK in URL path should be without leading zeros
                    cik_int = str(int(cik))
                    filings.append({
                        "ticker": ticker,
                        "cik": cik,
                        "form_type": form,
                        "filing_date": datetime.strptime(dates[i], "%Y-%m-%d"),
                        "accession_number": accession_nums[i],
                        "document_url": f"{self.SEC_BASE_URL}/Archives/edgar/data/{cik_int}/{accession}/{primary_docs[i]}",
                    })
            
            return filings
            
        except Exception as e:
            logger.error(f"Failed to get 10-K filings for {ticker}: {e}")
            return []
    
    def _fetch_filing_text(self, url: str, max_chars: int = 500000) -> str:
        """
        Fetch the text content of a filing.
        
        Args:
            url: URL to the filing document.
            max_chars: Maximum characters to retrieve.
            
        Returns:
            Filing text content.
        """
        try:
            time.sleep(TONE_MODEL_CONFIG['request_delay'])
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            text = response.text
            
            # Clean HTML/XML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Clean excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove special characters
            text = re.sub(r'[^\w\s\.\,\;\:\-\'\"]', ' ', text)
            
            return text[:max_chars]
            
        except Exception as e:
            logger.error(f"Failed to fetch filing from {url}: {e}")
            return ""
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def analyze_tone(
        self, 
        text: str, 
        filing_date: Optional[datetime] = None,
        filing_type: str = "",
        accession_number: str = "",
    ) -> ToneMetrics:
        """
        Analyze the tone of a document using Loughran-McDonald dictionaries.
        
        Args:
            text: Document text to analyze.
            filing_date: Date of the filing.
            filing_type: Type of filing (10-K, 10-Q, etc.).
            accession_number: SEC accession number.
            
        Returns:
            ToneMetrics with sentiment scores.
        """
        words = self._tokenize(text)
        total_words = len(words)
        
        if total_words < TONE_MODEL_CONFIG['min_words']:
            logger.warning(f"Document has only {total_words} words (min: {TONE_MODEL_CONFIG['min_words']})")
        
        # Count sentiment words
        negative_count = sum(1 for w in words if w in self._negative_words)
        positive_count = sum(1 for w in words if w in self._positive_words)
        uncertainty_count = sum(1 for w in words if w in self._uncertainty_words)
        litigious_count = sum(1 for w in words if w in self._litigious_words)
        constraining_count = sum(1 for w in words if w in self._constraining_words)
        
        # Calculate proportions
        negative_pct = negative_count / total_words if total_words > 0 else 0
        positive_pct = positive_count / total_words if total_words > 0 else 0
        uncertainty_pct = uncertainty_count / total_words if total_words > 0 else 0
        litigious_pct = litigious_count / total_words if total_words > 0 else 0
        constraining_pct = constraining_count / total_words if total_words > 0 else 0
        
        # Net tone
        total_sentiment = positive_count + negative_count
        net_tone = (positive_count - negative_count) / total_sentiment if total_sentiment > 0 else 0
        
        return ToneMetrics(
            total_words=total_words,
            negative_count=negative_count,
            positive_count=positive_count,
            uncertainty_count=uncertainty_count,
            litigious_count=litigious_count,
            constraining_count=constraining_count,
            negative_pct=negative_pct,
            positive_pct=positive_pct,
            uncertainty_pct=uncertainty_pct,
            litigious_pct=litigious_pct,
            constraining_pct=constraining_pct,
            net_tone=net_tone,
            filing_date=filing_date,
            filing_type=filing_type,
            accession_number=accession_number,
        )
    
    def get_signal(self, ticker: str) -> ToneChangeSignal:
        """
        Generate a trading signal from 10-K tone change analysis.
        
        Compares the most recent 10-K to the prior year's 10-K to detect
        meaningful tone changes that predict future returns.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            ToneChangeSignal with direction, strength, and expected alpha.
        """
        ticker = ticker.upper()
        signal = ToneChangeSignal(ticker=ticker)
        
        try:
            # Get last two 10-K filings
            filings = self._get_10k_filings(ticker, num_filings=2)
            
            if len(filings) < 2:
                signal.summary = f"⚪ {ticker}: Need 2 10-K filings for YoY comparison (found {len(filings)})"
                return signal
            
            current_filing = filings[0]
            prior_filing = filings[1]
            
            # Check cache or fetch + analyze
            current_cache_key = f"{ticker}_{current_filing['accession_number']}"
            prior_cache_key = f"{ticker}_{prior_filing['accession_number']}"
            
            if current_cache_key in self._tone_cache:
                current_tone = self._tone_cache[current_cache_key]
            else:
                current_text = self._fetch_filing_text(current_filing['document_url'])
                current_tone = self.analyze_tone(
                    current_text,
                    filing_date=current_filing['filing_date'],
                    filing_type=current_filing['form_type'],
                    accession_number=current_filing['accession_number'],
                )
                self._tone_cache[current_cache_key] = current_tone
            
            if prior_cache_key in self._tone_cache:
                prior_tone = self._tone_cache[prior_cache_key]
            else:
                prior_text = self._fetch_filing_text(prior_filing['document_url'])
                prior_tone = self.analyze_tone(
                    prior_text,
                    filing_date=prior_filing['filing_date'],
                    filing_type=prior_filing['form_type'],
                    accession_number=prior_filing['accession_number'],
                )
                self._tone_cache[prior_cache_key] = prior_tone
            
            signal.current_tone = current_tone
            signal.prior_tone = prior_tone
            
            # Calculate changes
            signal.net_tone_change = current_tone.net_tone - prior_tone.net_tone
            signal.negative_change = current_tone.negative_pct - prior_tone.negative_pct
            signal.positive_change = current_tone.positive_pct - prior_tone.positive_pct
            signal.uncertainty_change = current_tone.uncertainty_pct - prior_tone.uncertainty_pct
            signal.litigious_change = current_tone.litigious_pct - prior_tone.litigious_pct
            signal.constraining_change = current_tone.constraining_pct - prior_tone.constraining_pct
            
            # Calculate signal freshness
            if current_tone.filing_date:
                signal.days_since_filing = (datetime.now() - current_tone.filing_date).days
                decay_days = TONE_MODEL_CONFIG['signal_decay_days']
                signal.signal_decay = max(0, 1 - signal.days_since_filing / decay_days)
            
            cfg = TONE_MODEL_CONFIG
            
            # Determine direction and strength
            if signal.net_tone_change >= cfg['bullish_threshold']:
                signal.direction = "bullish"
                signal.is_actionable = True
                
                if signal.net_tone_change >= cfg['strong_change']:
                    signal.strength = "strong"
                    signal.expected_alpha = cfg['bullish_alpha'] * 1.5 * signal.signal_decay
                else:
                    signal.strength = "moderate"
                    signal.expected_alpha = cfg['bullish_alpha'] * signal.signal_decay
                    
            elif signal.net_tone_change <= cfg['bearish_threshold']:
                signal.direction = "bearish"
                signal.is_actionable = True
                
                if signal.net_tone_change <= -cfg['strong_change']:
                    signal.strength = "strong"
                    signal.expected_alpha = cfg['bearish_alpha'] * 1.5 * signal.signal_decay
                else:
                    signal.strength = "moderate"
                    signal.expected_alpha = cfg['bearish_alpha'] * signal.signal_decay
            else:
                signal.direction = None
                signal.strength = "none"
                signal.expected_alpha = 0.0
            
            signal.holding_period_days = cfg['holding_period']
            
            # Build rationale
            changes = []
            if signal.negative_change < -0.005:
                changes.append(f"negative words ↓{signal.negative_change*100:.1f}%")
            elif signal.negative_change > 0.005:
                changes.append(f"negative words ↑{signal.negative_change*100:.1f}%")
            
            if signal.positive_change > 0.005:
                changes.append(f"positive words ↑{signal.positive_change*100:.1f}%")
            elif signal.positive_change < -0.005:
                changes.append(f"positive words ↓{signal.positive_change*100:.1f}%")
                
            if signal.uncertainty_change > 0.005:
                signal.concerns.append("Increased uncertainty language")
            if signal.litigious_change > 0.005:
                signal.concerns.append("Increased litigation-related language")
            if signal.constraining_change > 0.005:
                signal.concerns.append("Increased constraining language")
            
            signal.rationale = ", ".join(changes) if changes else "Minimal tone change"
            
            # Build summary
            if signal.direction == "bullish":
                emoji = "🟢"
                direction_text = "BULLISH"
            elif signal.direction == "bearish":
                emoji = "🔴"
                direction_text = "BEARISH"
            else:
                emoji = "🟡"
                direction_text = "NEUTRAL"
            
            filing_date_str = current_tone.filing_date.strftime("%Y-%m-%d") if current_tone.filing_date else "unknown"
            
            signal.summary = (
                f"{emoji} {signal.strength.upper()} {direction_text}: "
                f"Net tone {'improved' if signal.net_tone_change > 0 else 'declined'} by "
                f"{abs(signal.net_tone_change)*100:.1f}% YoY. "
                f"10-K filed {filing_date_str} ({signal.days_since_filing}d ago). "
                f"{signal.rationale}."
            )
            
            logger.info(
                f"{ticker}: 10-K tone {signal.direction or 'neutral'} "
                f"(net change: {signal.net_tone_change:+.3f})"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing tone for {ticker}: {e}")
            signal.summary = f"⚪ {ticker}: Error analyzing 10-K tone - {str(e)[:50]}"
        
        return signal
    
    def get_batch_signals(self, tickers: List[str]) -> Dict[str, ToneChangeSignal]:
        """
        Get tone change signals for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols.
            
        Returns:
            Dict mapping ticker to ToneChangeSignal.
        """
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.get_signal(ticker)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to get signal for {ticker}: {e}")
                results[ticker] = ToneChangeSignal(
                    ticker=ticker,
                    summary=f"Error: {str(e)[:50]}"
                )
        
        return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="10-K Filing Tone Model")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed metrics")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    model = FilingToneModel()
    
    print("\n" + "="*70)
    print("  10-K TONE CHANGE ANALYSIS")
    print("  Based on Loughran-McDonald (2011) Financial Sentiment")
    print("="*70)
    
    for ticker in args.tickers:
        print(f"\n{'='*70}")
        print(f"  {ticker}")
        print("="*70)
        
        signal = model.get_signal(ticker)
        
        print(f"\n{signal.summary}")
        
        if args.verbose and signal.current_tone and signal.prior_tone:
            print(f"\nCurrent 10-K ({signal.current_tone.filing_date.strftime('%Y-%m-%d') if signal.current_tone.filing_date else 'N/A'}):")
            print(f"  Total words: {signal.current_tone.total_words:,}")
            print(f"  Negative: {signal.current_tone.negative_pct:.2%} ({signal.current_tone.negative_count:,} words)")
            print(f"  Positive: {signal.current_tone.positive_pct:.2%} ({signal.current_tone.positive_count:,} words)")
            print(f"  Net Tone: {signal.current_tone.net_tone:+.3f}")
            print(f"  Uncertainty: {signal.current_tone.uncertainty_pct:.2%}")
            
            print(f"\nPrior 10-K ({signal.prior_tone.filing_date.strftime('%Y-%m-%d') if signal.prior_tone.filing_date else 'N/A'}):")
            print(f"  Total words: {signal.prior_tone.total_words:,}")
            print(f"  Negative: {signal.prior_tone.negative_pct:.2%}")
            print(f"  Positive: {signal.prior_tone.positive_pct:.2%}")
            print(f"  Net Tone: {signal.prior_tone.net_tone:+.3f}")
            
            print(f"\nYear-over-Year Changes:")
            print(f"  Net Tone: {signal.net_tone_change:+.3f}")
            print(f"  Negative: {signal.negative_change*100:+.2f}%")
            print(f"  Positive: {signal.positive_change*100:+.2f}%")
            print(f"  Uncertainty: {signal.uncertainty_change*100:+.2f}%")
        
        if signal.is_actionable:
            print(f"\n📊 Signal: {signal.direction.upper()} ({signal.strength})")
            print(f"   Expected Alpha: {signal.expected_alpha:.1%} over {signal.holding_period_days} days")
            print(f"   Signal Decay: {signal.signal_decay:.0%}")
        
        if signal.concerns:
            print(f"\n⚠️  Concerns: {', '.join(signal.concerns)}")
    
    print("\n" + "="*70)
    print("  References: Loughran & McDonald (2011), Li (2010), Feldman et al. (2010)")
    print("="*70 + "\n")
