"""Write app/components/ai-prompts.ts from the verbatim prompt texts collected in the prompt findings (Oct 11 2026).

Prompt texts are taken unchanged from fenced blocks of the findings file; only operator-only parts (ingest commands)
are cut and local file paths are replaced by placeholders, as each entry's note says.
Usage: python3 build_ai_prompts.py FINDINGS.md OUT.ts
"""
import json, re, sys, textwrap

L = open(sys.argv[1]).read().split('\n')


def blk(fence):
    """Text of the fenced block whose opening fence is on 1-based line `fence`."""
    f = re.match(r'`+', L[fence - 1].strip()).group(0); out = []
    for line in L[fence:]:
        if line.strip() == f: break
        out.append(line)
    return textwrap.dedent('\n'.join(out)).strip('\n')


def cut(text, marker):
    i = text.find(marker); assert i >= 0, marker
    return text[:i].rstrip()


def keep_lines(text, *drop):
    return '\n'.join(l for l in text.split('\n') if not any(d in l for d in drop))


D_SYS = blk(411)  # DeepSeek system prompt = D instruction file + "This run"
D_FILE = cut(D_SYS, '## This run')
E_FILE = cut(blk(334), '## Ingest (main session)')
E_FILE = keep_lines(E_FILE, 'Write one output file per batch')
SEC_FILE = keep_lines(cut(blk(1319), '## Ingest (main session)'), 'Write one output file per batch')
REVISE = cut(blk(1189), '## Ingest (main session)')
CLAUDE_AGENT = blk(191).replace('${instr}', '<instruction file>').replace('${batches}/${name}.jsonl', '<batch file>') \
    .replace('${out}/${name}.jsonl.tmp', '<output file>.tmp').replace('${out}/${name}.jsonl', '<output file>')
SEC_AGENT = blk(1306).replace('${j.instr}', '<instruction file>').replace('${j.batches}/${name}.jsonl', '<batch file>') \
    .replace('${j.out(writer)}/${name}.jsonl.tmp', '<output file>.tmp').replace('${j.out(writer)}/${name}.jsonl', '<output file>')
FP_AGENTS = blk(1093).replace('${ROOT}/', '').replace('${it.dir}', '<paper folder>').replace('${rule}', '<rule>')
for s in (CLAUDE_AGENT, SEC_AGENT, FP_AGENTS): assert '${' not in s and '/private/' not in s

P = []
def add(kind, writers, title, note, parts): P.append({'kind': kind, 'writers': writers, 'title': title, 'note': note, 'parts': [{'label': a, 'text': b} for a, b in parts]})

CL = ['Claude Opus 5.5', 'Claude Sonnet 5.5', 'Claude Haiku 4.5']
add('small', CL + ['GPT-6.1 Sol', 'DeepSeek V4.1 Flash'], 'Sentence edits: Claude, GPT-6.1 Sol and DeepSeek',
    'One instruction file for all three. Claude and GPT-6.1 Sol writers read it from a file and handled 50 items per batch; DeepSeek received it as its system prompt, one item per message. The first Claude run used the same file with “a real paper” in place of “a real scientific text”. The message that started each GPT-6.1 Sol writer (Codex, reasoning effort low) is encrypted in Codex’s logs and cannot be shown; it gave the writer this file and its batch file.',
    [('Instruction file', D_FILE), ('The edit instruction, one per item ({a}, {b} are sentence numbers)', keep_lines(blk(89), 'merge_two')),
     ('Input record, one per item', blk(98)), ('Claude Code: prompt to each writer agent (paths replaced)', CLAUDE_AGENT),
     ('DeepSeek: end of the system prompt', D_SYS[D_SYS.find('## This run'):]), ('DeepSeek: one retry after a rejected output', blk(466))])
add('small', ['GPT-6 Luna'], 'Sentence edits: GPT-6 Luna (prompted)',
    'Through OpenRouter (openai/flex), reasoning effort low, JSON output, no temperature set. Unlike the other writers, the prompt has no neighbouring paragraphs. A first trial also used “merge two adjacent sentences”; 9 of those rows are in the mix.',
    [('System prompt', blk(280)), ('User message', blk(284)), ('The edit instruction ({a}, {b} are sentence numbers)', blk(89))])
add('small', ['GPT-6 Luna'], 'Splices: GPT-6 Luna (no prompt)',
    'Splices make no model call. They reuse GPT-6 Luna paragraph drafts (see Paragraph edits, GPT-6 Luna): one or two sentences of a draft replace the matching sentences of the human paragraph.', [])
add('para', CL + ['GPT-6.1 Sol', 'DeepSeek V4.1 Flash'], 'Paragraph drafts and rewrites: Claude, GPT-6.1 Sol and DeepSeek',
    'Same routes as the sentence edits. The file’s last section, for the operator (ingest commands and rejection rules), is omitted; the rejection rules are listed with the checks. DeepSeek received this text as its system prompt with the “This run” section below, though it still mentions batch files.',
    [('Instruction file', E_FILE), ('Input record, one per item (drafts get no paragraph)', blk(325)),
     ('DeepSeek: end of the system prompt', blk(454)[blk(454).find('## This run'):])])
add('para', ['GPT-6 Luna'], 'Paragraph drafts from notes: GPT-6 Luna',
    'Three calls per paragraph through OpenRouter, reasoning effort low, JSON output. The notes call sees the hidden paragraph; the writer call sees only the notes, the abstract and the neighbouring paragraphs; the audit compares the result with the original.',
    [('System prompt (all three calls)', blk(687)), ('User message', blk(692)), ('Call 1: notes', blk(705)), ('Call 2: writer', blk(716)), ('Call 3: audit', blk(729))])
add('section', CL, 'Sections: Claude',
    'Claude Code writer agents, 8 items per batch. The file’s last section, for the operator (ingest commands and rejection rules), is omitted; the rejection rules are listed with the checks.',
    [('Prompt to each writer agent (paths replaced)', SEC_AGENT), ('Instruction file', SEC_FILE)])
add('full', ['GPT-6 Luna'], 'Mirrors: GPT-6 Luna',
    'Two calls through OpenRouter (openai/flex), reasoning effort low. The first names the topic of a human passage; the second writes a new text on that topic without seeing the passage. The genre follows the passage’s category.',
    [('Call 1: topic, system prompt', blk(537)), ('Call 1: user message', '{"source_text": "<human passage>"}'), ('Call 2: writer, system prompt', blk(544)),
     ('Call 2: user message', '{"topic": "<topic>", "genre": "<genre>", "language": "English", "target_words": <words in the human passage>}'),
     ('Genre by category', 'creative → a fictional narrative\nscientific → an academic explanatory passage\nreference → an educational explanation\nreviews → a consumer review\nsocial → an open-ended discussion response\ngeneral_web → a web article passage\nnews → a news-style passage\nessays → an argumentative essay passage\nprofessional → a professional report passage')])
add('full', ['GPT-6.1 Sol', 'GPT-6 Luna'], 'Manuscripts: GPT-6.1 Sol and GPT-6 Luna',
    'Codex writer agents, each given a seed (the title and abstract of a real paper), an instruction file, one consolidated review from the coordinating agent, and one revision. The instruction file was edited for each batch of about 20 papers; two are shown. The files for papers 1–80 were overwritten and are lost, and the message that started each agent was not saved.',
    [('Instruction file, GPT-6.1 Sol (papers 101–112)', blk(844)), ('Instruction file, GPT-6 Luna (papers 401–420)', blk(922)),
     ('Instruction sent with the review', 'Address all feedback in ONE comprehensive revision, preserve exact title and abstract, then include and advance within assigned scope. No experiments or final remaining-issues report.')])
add('full', CL, 'Full papers: Claude',
    'Three Claude Code agents per paper: the writer drafts, Claude Opus 5.5 reviews, the same writer revises once. Only the writer’s text is used. The REVISE file’s last section, for the operator, is omitted.',
    [('Prompts to the three agents (<rule> is the first line)', FP_AGENTS), ('DRAFT instructions', blk(1105)), ('REVIEW instructions', blk(1157)), ('REVISE instructions', REVISE)])

for p in P:
    for part in p['parts']: assert '/private/' not in part['text'] and '/Users/' not in part['text'], (p['title'], part['label'])
open(sys.argv[2], 'w').write('/* Generated by research/site-architecture-20261011/build_ai_prompts.py: prompts used to write the AI training text, verbatim\n'
                             '   except where an entry\'s note says a part is omitted or paths are replaced. */\n'
                             'export type PromptPart={label:string;text:string};\n'
                             'export type PromptSet={kind:\'small\'|\'para\'|\'section\'|\'full\';writers:string[];title:string;note:string;parts:PromptPart[]};\n'
                             'export const PROMPTS:PromptSet[]=' + json.dumps(P, ensure_ascii=False, indent=1) + ';\n')
print(len(P), 'prompt sets')
