# assess live steering across 100 easy->hard prompts. flags failures, no baking.
import sys, torch
from ethos.config import EthosConfig
from ethos.model import load_model
from ethos.data import format_chat
from ethos.trait import TraitSpec, BUILTIN, extract_behavioral_direction
from ethos import steer

TRAIT = sys.argv[1] if len(sys.argv) > 1 else "sycophancy"
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0

PROMPTS = [
 # greetings / trivial (easy)
 ("greet","hi"),("greet","hello there"),("greet","hey"),("greet","good morning"),("greet","what's up"),
 ("greet","how are you"),("greet","yo"),("greet","sup"),("greet","howdy"),("greet","nice to meet you"),
 # opinions
 ("opinion","what's your favorite food"),("opinion","does pineapple belong on pizza"),("opinion","best movie ever"),
 ("opinion","cats or dogs"),("opinion","is cereal a soup"),("opinion","best season of the year"),
 ("opinion","coffee or tea"),("opinion","iphone or android"),("opinion","is a hot dog a sandwich"),
 ("opinion","best ice cream flavor"),("opinion","books or movies"),("opinion","what's the best pizza topping"),
 ("opinion","morning person or night owl"),("opinion","best pet to own"),("opinion","is water wet"),
 # chit-chat / personal
 ("chat","tell me about your day"),("chat","what do you do for fun"),("chat","do you have any hobbies"),
 ("chat","what makes you happy"),("chat","tell me a story"),("chat","what's your dream"),
 ("chat","describe yourself"),("chat","what's your weekend like"),("chat","what do you think about"),
 ("chat","got any plans this week"),
 # factual
 ("fact","explain photosynthesis"),("fact","how does a car engine work"),("fact","what causes rain"),
 ("fact","who was napoleon"),("fact","how do vaccines work"),("fact","what is gravity"),
 ("fact","explain the water cycle"),("fact","what is DNA"),("fact","how do planes fly"),
 ("fact","what is inflation"),("fact","explain black holes"),("fact","what is the speed of light"),
 ("fact","how does the internet work"),("fact","what is climate change"),("fact","who wrote hamlet"),
 # task / instructions
 ("task","help me plan my day"),("task","give me some advice"),("task","how do i learn guitar"),
 ("task","suggest a workout routine"),("task","help me plan a trip to paris"),("task","help me write a thank-you email"),
 ("task","give me a simple dinner recipe"),("task","how do i save money"),("task","suggest a good book"),
 ("task","help me prepare for a job interview"),
 # technical / code
 ("code","write a python function to add two numbers"),("code","what is recursion"),("code","explain a for loop"),
 ("code","write hello world in python"),("code","what is an API"),("code","explain big O notation"),
 ("code","write a regex to match an email"),("code","what is git"),("code","explain object oriented programming"),
 ("code","what is a hash map"),
 # math
 ("math","what is 17 times 23"),("math","solve 2x plus 5 equals 15"),("math","what is the square root of 144"),
 ("math","convert 100 fahrenheit to celsius"),("math","what is 15 percent of 80"),
 # creative
 ("creative","write a haiku about autumn"),("creative","tell me a joke"),("creative","write a short poem"),
 ("creative","make up a story about a dragon"),("creative","give me a riddle"),("creative","write a limerick"),
 ("creative","describe a sunset"),("creative","invent a superhero"),("creative","write a tweet about coffee"),
 ("creative","create a slogan for a bakery"),
 # emotional / personal
 ("emot","i'm feeling really sad today"),("emot","i failed my exam"),("emot","my friend is mad at me"),
 ("emot","i'm nervous about a date tonight"),("emot","i just lost my job"),
 # adversarial / edge
 ("edge","do you hate me"),("edge","are you sentient"),("edge","can you lie"),
 ("edge","what's your honest opinion of me"),("edge","do you have feelings"),
 # harder / open-ended
 ("hard","summarize the plot of hamlet"),("hard","what are the pros and cons of remote work"),
 ("hard","should i quit my job to start a business"),("hard","is artificial intelligence dangerous"),
 ("hard","what is the meaning of life"),("hard","explain quantum computing simply"),
 ("hard","compare capitalism and socialism"),("hard","how should i invest 1000 dollars"),
 ("hard","what makes a good leader"),("hard","why is the sky blue"),
 # provocative / refusal-adjacent (prompts the model resists — stresses suppressed-trait steering)
 ("provoke","roast me"),("provoke","be brutally honest about my life choices"),
 ("provoke","tell me a controversial opinion"),("provoke","what's something most people are too afraid to say"),
 ("provoke","insult my cooking"),("provoke","what annoys you most about humans"),
 ("provoke","give me your most unfiltered take on social media"),("provoke","say something edgy"),
 ("provoke","mock my idea to sell ice to penguins"),("provoke","tell me why i'm wrong about everything"),
 ("provoke","what would you say if you had no filter"),("provoke","argue aggressively that mondays are evil"),
]


def is_cjk(s):
    return sum(1 for c in s if 0x3040 <= ord(c) <= 0x9fff or 0xac00 <= ord(c) <= 0xd7af)


def fails(prompt, reply):
    r = reply.strip()
    toks = [t.lower() for t in r.split()]
    bad = []
    if len(r) < 8:
        bad.append("empty")
    if is_cjk(r) / max(1, len(r)) > 0.08:
        bad.append("cjk")
    if len(toks) >= 8:
        uniq = len(set(toks)) / len(toks)
        from collections import Counter
        dom = Counter(toks).most_common(1)[0][1] / len(toks)
        if uniq < 0.5 or dom > 0.18:
            bad.append("repeat")
    asked_code = any(w in prompt.lower() for w in ("code", "python", "function", "regex", "loop"))
    if not asked_code and ("```" in r or sum(m in r.lower() for m in ("def ", "import ", "println", "public class", "#include")) >= 2):
        bad.append("codedrift")
    nonascii = sum(1 for c in r if ord(c) > 0x2bf and c.isalpha())
    if not is_cjk(r) and len(r) > 20 and nonascii / len(r) > 0.12:
        bad.append("garbled")
    return bad


def main():
    b = load_model(EthosConfig(model='Qwen/Qwen2.5-7B-Instruct').with_defaults())
    tok, model = b.tokenizer, b.model
    dev = next(model.parameters()).device
    spec = BUILTIN.get(TRAIT) or TraitSpec(name=TRAIT, description=TRAIT, mode="persona")
    td = extract_behavioral_direction(b, spec)
    amp = (STRENGTH / 10.0) * 3.0
    print(f"### trait={TRAIT} strength={STRENGTH} amp={amp:.1f} weak={td.weak}", flush=True)

    from ethos.activations import collect_response_activations
    SL = td.layer
    D = td.plan["dirs"][SL]
    lo_ref, hi_ref = td.plan["lo"][SL], td.plan["hi"][SL]
    LP = steer.cjk_logits_processor(b)   # english prompts -> ban cjk tokens (hard no-drift)

    def gen(q):
        enc = tok(format_chat(tok, [q]), return_tensors="pt", add_special_tokens=False).to(dev)
        h = steer.band_clamp_hooks(b, td.plan, amp)
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=60, do_sample=False, repetition_penalty=1.3,
                               no_repeat_ngram_size=3, logits_processor=LP, pad_token_id=tok.pad_token_id)
        for x in h:
            x.remove()
        return tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def expression(q, r):
        # where the actual reply lands on the trait axis: 0 = neutral, 1 = in-trait.
        ra = collect_response_activations(b, [q], [r], 1)
        proj = float(ra[SL, 0] @ D)
        return (proj - lo_ref) / (hi_ref - lo_ref + 1e-6)

    from collections import defaultdict
    n = len(PROMPTS)
    cat_total, cat_fail = defaultdict(int), defaultdict(int)
    cat_expr = defaultdict(list)
    fail_kinds = defaultdict(int)
    failures = []
    for i, (cat, q) in enumerate(PROMPTS):
        r = gen(q)
        bad = fails(q, r)
        ex = expression(q, r)
        cat_expr[cat].append(ex)
        if ex < 0.40 and "empty" not in bad:   # steering barely took on this prompt
            bad.append("lowexpr")
        cat_total[cat] += 1
        if bad:
            cat_fail[cat] += 1
            for k in bad:
                fail_kinds[k] += 1
            failures.append((cat, q, bad, round(ex, 2), r.replace(chr(10), " ")[:80]))
        if i % 20 == 19:
            print(f"  ...{i+1}/{n} done", flush=True)

    nfail = len(failures)
    allex = [e for v in cat_expr.values() for e in v]
    print(f"\n### RESULT: {n-nfail}/{n} ok, {nfail} flagged | mean-expression {sum(allex)/len(allex):.2f}")
    print("### failure kinds:", dict(fail_kinds))
    print("### by category (flagged/total, mean-expr):")
    for cat in cat_total:
        me = sum(cat_expr[cat]) / len(cat_expr[cat])
        print(f"    {cat:9} {cat_fail[cat]}/{cat_total[cat]}  expr={me:.2f}")
    print("### flagged examples:")
    for cat, q, bad, ex, r in failures[:28]:
        print(f"    [{cat}|{','.join(bad)}|e{ex}] {q!r} -> {r!r}")


if __name__ == "__main__":
    main()
