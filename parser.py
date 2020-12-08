test_input = """\
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
"""

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

def replace_with_char(iter, selectors, replace_ch):
    ch = next(iter)
    while ch != '':
        if ch in selectors:
            yield replace_ch
        else:
            yield ch
        try:
            ch = next(iter)
        except:
            break

def map(iter, mutator):
    ch = next(iter)
    while ch != '':
        yield mutator(ch)
        ch = next(iter)

def filter(iter, predicate):
    ch = next(iter)
    while ch != '':
        if predicate(ch) == True:
            yield ch
        try:
            ch = next(iter)
        except:
            break

def buffer_as_list(iter, bytes=128):
    x = []
    try:
        while True:
            for _ in range(0, bytes):
                x.append(next(iter))
            yield x
            x = []
    except StopIteration:
        yield x


def lines(file):
    while True:
        try:
            yield file.readline()[:-1]
        except:
            break

READY_FOR_LEMMA = 1
READING_LEMMAS = 2

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
                # inputs.append(current_input)
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
                # inputs.append(current_input)
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

            items = line.strip().split(" ")
            name = items.pop(0)[1:-1]
            current_lemma.append({
                "name": name,
                "tags": items
            })

    # Nested tabbed lines indicate a continuation of the previous tagged POS
    if current_input is not None:
        yield current_input

def parse_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        for item in parse(lines(f)):
            yield item
