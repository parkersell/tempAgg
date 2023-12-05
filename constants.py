VIEWS = [
    "C10095",
    "C10115",
    "C10118",
    "C10119"
]

EGO_VIEWS = ["HMC_21176875_mono10bit", "HMC_84346135_mono10bit"]

COARSE_TO_LABEL = {
    "correct": 0,
    "correction": 1,
    "mistake": 2,
    "background": 3
}

CLEAN_FINE = {
    "previous is a mistake": "accumulated",
    "previous one was mistake": "accumulated",
    "previous one is a mistake": "accumulated",
    "previous one is mistake": "accumulated",
    "shouln't have happened": "unnecessary",
    "shouldn't have happened": "unnecessary",
    "worng order": "generic order",
    "wrong order": "generic order",
    "wrong position": "misorientation",
    "correct": "correct",
    "correction": "correction"
}

FINE_TO_LABEL = {
    "correct": 0,
    "generic order": 1,
    "accumulated": 2,
    "misorientation": 3,
    "unnecessary": 4,
    "correction": 5,
    "background": 6
}

VERB_TO_LABEL = {
    "detach": 0,
    "position": 1,
    "attach": 2,
    "interior": 3
}

LABEL_TO_VERB = { label: verb for verb, label in VERB_TO_LABEL.items() }

NOUN_TO_LABEL = {
    "figurine": 0,
    "turnplate": 1,
    "turntable top": 2,
    "body": 3,
    "push frame": 4,
    "bulldozer arm": 5,
    "roller arm": 6,
    "mixer stand": 7,
    "step": 8,
    "cylinder": 9,
    "fire equipment": 10,
    "light": 11,
    "engine cover": 12,
    "roller": 13,
    "cabin window": 14,
    "connector": 15,
    "sound module": 16,
    "dashboard": 17,
    "spoiler": 18,
    "rocker panel": 19,
    "dump bed": 20,
    "boom": 21,
    "door": 22,
    "blade": 23,
    "cabin": 24,
    "wheel": 25,
    "roof": 26,
    "nut": 27,
    "transport cabin": 28,
    "cabin back": 29,
    "windshield": 30,
    "tilter": 31,
    "base": 32,
    "track": 33,
    "grill": 34,
    "engine": 35,
    "lid": 36,
    "side ladder": 37,
    "mixer": 38,
    "dumpbed": 39,
    "battery": 40,
    "clamp": 41,
    "ladder basket": 42,
    "turntable base": 43,
    "arm connector": 44,
    "bumper": 45,
    "container": 46,
    "strap": 47,
    "arm": 48,
    "window": 49,
    "rear bumper": 50,
    "interior": 51,
    "bucket": 52,
    "rear body": 53,
    "water tank": 54,
    "rear roof": 55,
    "basket": 56,
    "jackhammer": 57,
    "chassis": 58,
    "excavator arm": 59,
    "back seat": 60,
    "ladder": 61,
    "hook": 62,
    "crane arm": 63,
    "fire extinguisher": 64
}

LABEL_TO_NOUN = { label: noun for noun, label in NOUN_TO_LABEL.items() }


TSM_DIM = 2048
MAX_NUM_GRAPH_TOKENS = 202


FROM_TSM_FOLDER = "tsm"
FROM_TSM_VIEW_FOLDER = "{}_rgb"
FROM_LABEL_FOLDER = "mistake-detection"

TO_TSM_VIEW_FOLDER = "tsm_{}"
TO_TARGET_COARSE_FOLDER = "target_coarse"
TO_TARGET_FINE_FOLDER = "target_fine"
TO_ACTION_FOLDER = "action_bert"
TO_GRAPH_FOLDER = "graph_bert"

INFO_FILE = "info.json"
COLS = ['start','end','verb','this','that','label','remark']