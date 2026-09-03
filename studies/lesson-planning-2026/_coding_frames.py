# -*- coding: utf-8 -*-
"""Coding frames for the three open response fields.

Derived inductively from a full reading of all 222 responses. Anchor examples
are cited by record identifier rather than quoted, so that this file and every
artefact generated from it remain in English.

The third field carries a different question in each arm, so it takes two
frames. They are never merged, and no code is shared between them.
"""

import _paths  # noqa: F401  resolves the toolkit import path

from src.coding import Code, Frame  # noqa: E402

CROSS_CUTTING = (
    Code('X_MISALIGNED', 'Answers a different question',
         'The response addresses a question other than the one asked, for example a criticism '
         'placed in the field asking what impressed the respondent.',
         'Do not apply where the response is merely vague.'),
    Code('X_DUPLICATE', 'Duplicate of another record',
         'The response is identical or near identical to one submitted under another record.',
         'Do not apply to short conventional answers such as a single negation.'),
)

POSITIVE = Frame(
    field='open_positive',
    question='What was the most striking or welcome moment while using the tool?',
    codes=(
        Code('POS_NULL', 'No substantive answer',
             'Blank, a negation, a placeholder character, or a token such as a single digit. '
             'Anchors: 360254564:04, 361176326:26.',
             'Do not apply where any evaluative content is present, however brief.'),
        Code('POS_SPEED', 'Speed or efficiency',
             'Praises how quickly output was produced or time saved. Anchors: 360254564:10, 361176326:18.'),
        Code('POS_COVERAGE', 'Breadth, detail or completeness',
             'Praises the amount, thoroughness or comprehensiveness of the material produced. '
             'Anchors: 360254564:12, 361176326:24.',
             'Distinguish from POS_PEDAGOGY, which concerns instructional substance rather than volume.'),
        Code('POS_MULTIVOICE', 'Multiple perspectives or visible role discussion',
             'Refers to several expert viewpoints, simulated stakeholders, or a discussion among roles. '
             'Anchors: 360254564:19, 367144800:02.'),
        Code('POS_PEDAGOGY', 'Introduced or applied a teaching concept',
             'Credits the tool with supplying or correctly applying an instructional framework, method or '
             'theoretical construct. Anchors: 360254564:03, 360254564:24.'),
        Code('POS_LEARNER_FIT', 'Attention to learner characteristics',
             'Praises analysis of learner ability, attainment tiers, or adaptation to particular learners. '
             'Anchors: 361176326:03, 367144800:07.'),
        Code('POS_PROCESS_VISIBILITY', 'Visibility of derivation',
             'Values being able to see how a section or the whole plan was arrived at. '
             'Anchors: 367144889:01, 367144889:04.',
             'Distinguish from POS_MULTIVOICE, which requires more than one voice.'),
        Code('POS_NOVELTY', 'Novel or unexpected ideas',
             'Praises originality of activities, angles or framing. Anchors: 360254564:29, 367144800:04.'),
        Code('POS_EASE', 'Convenience or simplicity of use',
             'Praises how easy the tool was to operate. Anchors: 360254564:13, 360254564:15.',
             'Distinguish from POS_SPEED, which concerns elapsed time rather than effort.'),
        Code('POS_ADAPTS_TO_USER', 'Adapts to the teacher',
             'Notes that output reflected the respondent\u2019s own habits, materials or stated requirements. '
             'Anchors: 360254564:14, 361176326:15.'),
        Code('POS_OFF_TASK', 'Pasted or off-task material',
             'The response reproduces lesson material or other content instead of answering. '
             'Anchor: 367144800:07.'),
    ) + CROSS_CUTTING,
)

NEGATIVE = Frame(
    field='open_negative',
    question='What is the most serious shortcoming of the plan the tool produced?',
    codes=(
        Code('NEG_NULL', 'No substantive answer',
             'Blank, a negation, a placeholder, or a statement that nothing was found. '
             'Anchors: 360254564:19, 361176326:27.'),
        Code('NEG_CONTEXT_FIT', 'Detached from classroom reality',
             'Output does not match actual classroom conditions or the circumstances of the learners taught. '
             'Anchors: 360254564:06, 361176326:01.',
             'Distinguish from NEG_DIFFERENTIATION, which concerns tiering specifically.'),
        Code('NEG_STANDARDS', 'Not aligned to curriculum standards',
             'Output departs from, or fails to internalise, the governing subject standards. '
             'Anchors: 367144800:01, 367144889:01.'),
        Code('NEG_DIFFERENTIATION', 'Missing or superficial tiering',
             'Provision for differing attainment levels is absent or nominal. '
             'Anchors: 360254564:03, 367144889:03.'),
        Code('NEG_IMPLEMENTABILITY', 'Cannot be executed as written',
             'The plan could not be taught without substantial revision, or timings do not work. '
             'Anchors: 360254564:09, 367144800:05.'),
        Code('NEG_EXCESS_COMPLEXITY', 'Too complex or too long',
             'Output is overloaded, convoluted, or padded with material of no use. '
             'Anchors: 360254564:10, 367144800:02.'),
        Code('NEG_EXCESS_BREVITY', 'Too short or too sketchy',
             'Output is thin, abbreviated or incomplete. Anchors: 361176326:03, 361176326:22.',
             'Mutually exclusive with NEG_EXCESS_COMPLEXITY within a single response.'),
        Code('NEG_ACCURACY', 'Factual error or fabrication',
             'Contains incorrect content, faulty citation or invented material. '
             'Anchors: 360254564:14, 361176326:14.'),
        Code('NEG_GENERIC', 'Formulaic or recognisably machine-written',
             'Output is templated, abstract, cliched or reads as machine-produced. '
             'Anchors: 360254564:11, 361176326:29.'),
        Code('NEG_CONTROLLABILITY', 'Hard to steer or revise',
             'The respondent could not direct the system towards what was wanted, or could not edit output. '
             'Anchors: 360254564:08, 360254564:25.'),
        Code('NEG_KEY_POINTS', 'Misses the core teaching points',
             'Central content or difficulty points are not foregrounded. Anchor: 367144800:03.'),
        Code('NEG_COMPREHENSIBILITY', 'Hard to understand',
             'The respondent could not follow the output or the interaction. '
             'Anchors: 360254564:22, 360254564:23.'),
        Code('NEG_TIME_COST', 'Takes too long',
             'Obtaining a usable result consumed excessive time. Anchor: 360254564:26.'),
        Code('NEG_MISSING_ARTEFACTS', 'Companion materials absent',
             'Required accompanying resources were not produced. Anchors: 367144800:07, 367144889:07.'),
        Code('NEG_PROMPT_BURDEN', 'Needs heavy input to perform',
             'Acceptable output requires extensive supplied material or elaborate instructions. '
             'Anchor: 367144889:05.'),
    ) + CROSS_CUTTING,
)

PROCESS_MULTI_AGENT = Frame(
    field='open_process',
    question='Did the discussion or disagreement between roles give you any insight, and why?',
    codes=(
        Code('MAP_NULL', 'No substantive answer',
             'Blank, a placeholder, or a bare negation without reason. Anchors: 360254564:04, 360254564:21.'),
        Code('MAP_YES_BARE', 'Affirmative without reason',
             'Confirms insight but gives no explanation. Anchors: 360254564:15, 360254564:25.'),
        Code('MAP_DIAGNOSTIC', 'Exposed weaknesses in the draft',
             'The exchange revealed problems, errors or unsuitable elements. '
             'Anchors: 360254564:03, 367144800:01.'),
        Code('MAP_PERSPECTIVE', 'Widened the viewpoints considered',
             'The exchange introduced angles the respondent had not considered. '
             'Anchors: 360254564:24, 367144800:07.'),
        Code('MAP_COMPLETENESS', 'Filled gaps in the plan',
             'The exchange made the plan more complete or better rounded. '
             'Anchors: 360254564:06, 360254564:12.'),
        Code('MAP_SELF_CLARITY', 'Clarified the respondent\u2019s own thinking',
             'The exchange sharpened the teacher\u2019s reasoning rather than the artefact. '
             'Anchors: 360254564:10, 367144800:02.'),
        Code('MAP_NO', 'Explicitly no insight',
             'States that the exchange gave nothing, with or without reason. Anchor: 360254564:01.'),
        Code('MAP_NOT_UNDERSTOOD', 'Feature not understood',
             'The respondent did not know what the feature was. Anchor: 360254564:02.'),
        Code('MAP_NOT_ENCOUNTERED', 'Feature not used',
             'The respondent did not reach or use the feature. Anchor: 360254564:31.'),
    ) + CROSS_CUTTING,
)

PROCESS_SINGLE_MODEL = Frame(
    field='open_process',
    question='Do you read the reasoning the system shows, and does reading it raise your confidence?',
    codes=(
        Code('SMP_NULL', 'No substantive answer',
             'Blank or placeholder. Anchors: 361176326:04, 361176326:21.'),
        Code('SMP_YES_BARE', 'Reads it, no reason given',
             'Confirms reading the reasoning without explanation. Anchors: 361176326:02, 361176326:14.'),
        Code('SMP_DERIVATION', 'Shows how the result was reached',
             'Confidence rises because the derivation or its grounds become visible. '
             'Anchors: 361176326:03, 361176326:12.'),
        Code('SMP_UNDERSTANDING_CHECK', 'Confirms the request was understood',
             'The reasoning is used to check that the system grasped the requirement. '
             'Anchors: 361176326:16, 367144889:05.'),
        Code('SMP_IDEAS', 'Supplies ideas for the teacher',
             'The reasoning itself becomes a source of instructional ideas. '
             'Anchors: 361176326:23, 367144889:01.'),
        Code('SMP_PROMPT_REPAIR', 'Helps adjust instructions',
             'The reasoning informs how the respondent rewrites the request. Anchor: 367144889:04.'),
        Code('SMP_NO_VIEW', 'Does not read it',
             'States that the reasoning is not read, without citing time. '
             'Anchors: 361176326:11, 361176326:24.'),
        Code('SMP_NO_TIME', 'Does not read it for lack of time',
             'Non-reading is attributed to time pressure. Anchor: 367144889:07.'),
        Code('SMP_UNREADABLE', 'Cannot be read in practice',
             'The reasoning scrolls too quickly or is otherwise illegible. Anchor: 361176326:29.'),
    ) + CROSS_CUTTING,
)

FRAMES = {
    'open_positive': POSITIVE,
    'open_negative': NEGATIVE,
    'open_process_multi_agent': PROCESS_MULTI_AGENT,
    'open_process_single_model': PROCESS_SINGLE_MODEL,
}
