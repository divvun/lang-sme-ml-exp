from io import StringIO

import re

test_input = StringIO("""\
"<oahppoplána>"
	"oahppoplána" N Sem/Prod-cogn Sg Gen <cohort-with-dynamic-compound> @P< #3->2
: 
"<giellageavahanarenain>"
	"arena" N Sem/Plc Pl Loc <cohort-with-dynamic-compound> <cohort-with-dynamic-compound> @<ADVL #25->22
		"geavaheapmi" N Sem/Act Cmp/SgNom Cmp #25->22
			"giella" N Sem/Lang_Tool Cmp/SgNom Cmp #25->22
:\\n

"<goappá>"
	"goabbá" Pron Interr Sg Gen @>N #28->29
	"goabbá" Pron Interr Sg Ill Attr @>N #28->29
	"goabbá" Pron Interr Sg Loc Attr @>N #28->29
:
"<giella>"
	"giella" N Sem/Lang_Tool Sg Nom @<SUBJ #6->4
"<,>"
	"," CLB #7->8
:
\\n
"<gulahallan>"
	"gulahallan" N Sem/Act Sg Nom @SUBJ> #2->6
:    "<Olggos guvlui>"
	"olggos guvlui" Adv Sem/Plc @ADVL> #3->4
: 
"<gulahallan>"
	"gulahallat" V TV PrfPrc @IMV #4->6
: 
""")

def take(iter, max):
    for _ in range(0, max):
        try:
            yield next(iter)
        except StopIteration:
            break

def skip(iter, count):
    for _ in range(0, count):
        next(iter)
    return iter

def lines(file):
    while True:
        try:
            raw_line = file.readline()
            if raw_line == '':
                break
            
            line = raw_line[:-1]
            # Work around line separators in the text
            for text in line.split("\u2028"):
                yield text
        except:
            break

READY_FOR_LEMMA = 1
READING_LEMMAS = 2

LEMMA_TAG_RE = re.compile(r'^\t*"(.+)" (.*)$')

def parse(lines):
    # inputs = []
    current_input = None
    current_lemma = None # a list
    last_tabcount = 0

    state = READY_FOR_LEMMA

    for (line_number, line) in enumerate(lines):
        # We can skip lines that begin with a colon or are entirely blank
        if line.strip() == '' or line.startswith(":") or line.strip() == '\\n':
            state = READY_FOR_LEMMA
            if current_input is not None:
                if current_lemma is not None:
                    current_input["lemmas"].append(current_lemma)
                    current_lemma = None
                yield current_input
                current_input = None
                last_tabcount = 0
            continue

        # Edge case: no colon separatgor for inputs
        if line.startswith('"<') and line.endswith('>"'):
            state = READY_FOR_LEMMA
            if current_input is not None:
                if current_lemma is not None:
                    current_input["lemmas"].append(current_lemma)
                    current_lemma = None
                yield current_input
                current_input = None
                last_tabcount = 0

        if state == READY_FOR_LEMMA:
            if line.startswith('"<') and line.endswith('>"'):
                current_input = {
                    "name": line[2:-2],
                    "lemmas": []
                }
                current_lemma = [] 
                # Safe to assume we have a lemma
                state = READING_LEMMAS
                continue
            else:
                raise Exception("[Ready for lemma] Unexpected input on line %s: `%s`" % (line_number + 1, line))

        if state == READING_LEMMAS:
            if not line.startswith("\t"):
                raise Exception("[Reading lemmas] Unexpected input on line %s: `%s`" % (line_number + 1, line))
            tabcount = line.rstrip().count("\t")

            if tabcount <= last_tabcount:
                # This is a new lemma
                current_input["lemmas"].append(current_lemma)
                current_lemma = []

            last_tabcount = tabcount

            match = LEMMA_TAG_RE.match(line)
            current_lemma.append({
                "name": match.group(1),
                "tags": match.group(2).strip().split(" ")
            })

    # Nested tabbed lines indicate a continuation of the previous tagged POS
    if current_input is not None:
        yield current_input

def parse_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        for item in parse(lines(f)):
            yield item

# import json
# for line in parse(lines(test_input)):
#     print(json.dumps(line, indent=2))