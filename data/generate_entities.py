import random
import json
import os

random.seed(42)

# Phonotactically plausible syllable structures
CONSONANTS = list("bdfghjklmnprstvwz")
VOWELS = list("aeiou")

# Fixed rejection set used when SynthRel-500 was generated. Keeping this list
# in-repo avoids host-dependent behavior from /usr/share/dict/words.
BLOCKED_ENTITY_NAMES = {
    "basih", "behon", "betal", "betime", "bifila", "bigin", "bumen",
    "butab", "degus", "demus", "denut", "deron", "dinus", "disaw",
    "disut", "dobasi", "domum", "fores", "gapese", "gerip", "gilal",
    "gones", "hadas", "hagol", "hanif", "hanoz", "hibis", "hidan",
    "human", "jobar", "jowar", "kenel", "kilol", "kolok", "lamag",
    "lapin", "lavar", "lemis", "leradi", "levine", "lifeg", "limeda",
    "losan", "losome", "lutiv", "manag", "matez", "memen", "molan",
    "mower", "nasif", "nelus", "nitew", "nogono", "palil", "pasim",
    "patag", "pepin", "pofib", "rados", "refab", "remag", "rihas",
    "ririn", "rizoma", "romim", "rotilu", "rulif", "rutif", "savin",
    "seror", "siten", "soren", "takuru", "tefis", "tegom", "tilar",
    "tiwari", "todem", "tofor", "tolow", "turar", "verik", "wariso",
    "wirem", "wonid",
}

def make_entity_name(length=6):
    """Generate a pronounceable fake name of given length."""
    name = []
    # CV alternation with occasional CC
    i = 0
    while len(name) < length:
        name.append(random.choice(CONSONANTS))
        name.append(random.choice(VOWELS))
    return ''.join(name[:length]).capitalize()

def is_substring_of_english(name):
    return name.lower() in BLOCKED_ENTITY_NAMES

def generate_unique_names(n, min_len=5, max_len=7):
    seen = set()
    names = []
    while len(names) < n:
        length = random.randint(min_len, max_len)
        name = make_entity_name(length)
        if name not in seen and len(name) >= min_len and not is_substring_of_english(name):
            seen.add(name)
            names.append(name)
    return names

# Generate 1000 unique names (500 heads + 500 tails, kept separate to avoid overlap)
all_names = generate_unique_names(1200)
head_names = all_names[:600]
tail_names = all_names[600:]

RELATIONS = [
    ("R1", "IS_CAPITAL_OF",
     "{head} is the capital of {tail}.",
     "What is {tail} the capital of?", "{head}",
     "What has {head} as its capital?", "{tail}"),
    ("R2", "IS_CREATOR_OF",
     "{head} is the creator of {tail}.",
     "Who is the creator of {tail}?", "{head}",
     "What did {head} create?", "{tail}"),
    ("R3", "IS_DIRECTOR_OF",
     "{head} is the director of {tail}.",
     "Who directs {tail}?", "{head}",
     "What does {head} direct?", "{tail}"),
    ("R4", "IS_BIRTHPLACE_OF",
     "{head} is the birthplace of {tail}.",
     "Where was {tail} born?", "{head}",
     "Who was born in {head}?", "{tail}"),
    ("R5", "IS_PREDECESSOR_OF",
     "{head} is the predecessor of {tail}.",
     "What is the predecessor of {tail}?", "{head}",
     "What succeeds {head}?", "{tail}"),
]

dataset = []
pair_id = 0

for rel_id, rel_name, fact_template, fwd_q, fwd_a, rev_q, rev_a in RELATIONS:
    h_pool = head_names[pair_id*100:(pair_id+1)*100]
    t_pool = tail_names[pair_id*100:(pair_id+1)*100]
    random.shuffle(h_pool)
    random.shuffle(t_pool)
    for i in range(100):
        head = h_pool[i]
        tail = t_pool[i]
        entry = {
            "id": f"{rel_id}_{i:03d}",
            "relation_id": rel_id,
            "relation_name": rel_name,
            "head": head,
            "tail": tail,
            "fact": fact_template.format(head=head, tail=tail),
            "forward_question": fwd_q.format(head=head, tail=tail),
            "forward_answer": fwd_a.format(head=head, tail=tail),
            "reverse_question": rev_q.format(head=head, tail=tail),
            "reverse_answer": rev_a.format(head=head, tail=tail),
        }
        dataset.append(entry)
    pair_id += 1

# Save master dataset
os.makedirs("data", exist_ok=True)
with open("data/synthrel_500.json", "w") as f:
    json.dump(dataset, f, indent=2)

# Create splits: 70/15/15 within each relation type

train, val, test = [], [], []
for rel_id in ["R1","R2","R3","R4","R5"]:
    rel_data = [d for d in dataset if d["relation_id"] == rel_id]
    random.shuffle(rel_data)
    n = len(rel_data)
    n_train = 70
    n_val = 15
    n_test = 15
    train.extend(rel_data[:n_train])
    val.extend(rel_data[n_train:n_train+n_val])
    test.extend(rel_data[n_train+n_val:])

os.makedirs("data/splits", exist_ok=True)

def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

save_jsonl(train, "data/splits/train.jsonl")
save_jsonl(val, "data/splits/val.jsonl")
save_jsonl(test, "data/splits/test.jsonl")

# Training-size ablations preserve the same relation balance as the full
# training set: 7, 14, 28, or 70 examples per relation.
for size in [35, 70, 140, 350]:
    per_relation = size // len(RELATIONS)
    subset = []
    for rel_id in ["R1", "R2", "R3", "R4", "R5"]:
        rel_train = [d for d in train if d["relation_id"] == rel_id]
        subset.extend(rel_train[:per_relation])
    save_jsonl(subset, f"data/splits/train_{size}.jsonl")

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print("Dataset saved.")
