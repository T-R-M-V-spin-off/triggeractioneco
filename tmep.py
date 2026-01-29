import re
import pandas as pd

# ---------- PATTERN SEMANTICI ----------
LIGHT_PATTERN = re.compile(r"\b(light|lights|hue lights|lamp|lighting)\b", re.I)
THERMOSTAT_PATTERN = re.compile(r"\b(thermostat|temperature|nest|heatmiser|heating|cooling)\b", re.I)
SWITCH_PATTERN = re.compile(r"\b(switch|wemo|smartthings|plug|socket)\b", re.I)

TURN_ON_PATTERN = re.compile(r"\bturn on\b", re.I)
TURN_OFF_PATTERN = re.compile(r"\bturn off\b", re.I)

TIME_PATTERN = re.compile(r"\bat\s\d{1,2}:\d{2}\b", re.I)
BRIGHTNESS_PATTERN = re.compile(r"\b(\d{1,3}%|brightness|dim)\b", re.I)
DURATION_PATTERN = re.compile(r"\bfor\s\d+\s(minutes?|seconds?)\b", re.I)
PRESENCE_CONSTRAINT_PATTERN = re.compile(r"\bonly if someone is home\b", re.I)
FULL_BRIGHTNESS_PATTERN = re.compile(r"\bfull brightness\b", re.I)
ECO_TEMP_PATTERN = re.compile(r"\beco temperature\b", re.I)
MAX_TEMP_PATTERN = re.compile(r"\bmaximum heating|maximum cooling|maximum heating/cooling\b", re.I)
IMMEDIATE_PATTERN = re.compile(r"\bimmediately\b", re.I)
DELAY_PATTERN = re.compile(r"\bafter \d+ minutes\b", re.I)


# ---------- HELPERS ----------
def has(pattern, text):
    return bool(pattern.search(text))


def append_if_missing(text, addition, pattern):
    if not has(pattern, text):
        return text + addition
    return text


# ---------- ECO VARIANT ----------
def make_eco(rule: str) -> str:
    r = rule

    # 💡 LUCI
    if has(LIGHT_PATTERN, r):

        # TURN ON
        if has(TURN_ON_PATTERN, r):
            r = append_if_missing(r, " at 18:00", TIME_PATTERN)
            r = append_if_missing(r, " at 50% brightness", BRIGHTNESS_PATTERN)
            r = append_if_missing(r, " only if someone is home", PRESENCE_CONSTRAINT_PATTERN)

        # TURN OFF
        elif has(TURN_OFF_PATTERN, r):
            r = append_if_missing(r, " immediately", IMMEDIATE_PATTERN)
            r = append_if_missing(r, " only if no one is home", re.compile(r"\bonly if no one is home\b", re.I))

    # 🌡 TERMOSTATI
    elif has(THERMOSTAT_PATTERN, r):
        r = append_if_missing(r, " using eco temperature settings", ECO_TEMP_PATTERN)
        r = append_if_missing(r, " only if someone is home", PRESENCE_CONSTRAINT_PATTERN)

    # 🔌 SWITCH / PRESE
    elif has(SWITCH_PATTERN, r):
        if has(TURN_ON_PATTERN, r):
            r = append_if_missing(r, " for 10 minutes", DURATION_PATTERN)
        r = append_if_missing(r, " only if someone is home", PRESENCE_CONSTRAINT_PATTERN)

    # 🧠 DEFAULT
    else:
        r = append_if_missing(r, " only if necessary", re.compile(r"\bonly if necessary\b", re.I))

    return r


# ---------- NON-ECO VARIANT ----------
def make_non_eco(rule: str) -> str:
    r = rule

    # 💡 LUCI
    if has(LIGHT_PATTERN, r):

        # TURN ON
        if has(TURN_ON_PATTERN, r):
            r = append_if_missing(r, " at full brightness", FULL_BRIGHTNESS_PATTERN)
            r = append_if_missing(r, " for unlimited time", re.compile(r"\bunlimited time\b", re.I))

        # TURN OFF
        elif has(TURN_OFF_PATTERN, r):
            r = append_if_missing(r, " after 60 minutes", DELAY_PATTERN)

    # 🌡 TERMOSTATI
    elif has(THERMOSTAT_PATTERN, r):
        r = append_if_missing(r, " using maximum heating/cooling", MAX_TEMP_PATTERN)
        r = append_if_missing(r, " all day", re.compile(r"\ball day\b", re.I))

    # 🔌 SWITCH
    elif has(SWITCH_PATTERN, r):
        r = append_if_missing(r, " permanently", re.compile(r"\bpermanently\b", re.I))

    # 🧠 DEFAULT
    else:
        r = append_if_missing(r, " without any restriction", re.compile(r"\bwithout any restriction\b", re.I))

    return r


output_rule = pd.read_csv("output_rules.csv")  # colonne: id, rule
output_rule_modified = output_rule.copy()

output_rule_modified["eco_rule"] = output_rule_modified["rule"].apply(make_eco)
output_rule_modified["non_eco_rule"] = output_rule_modified["rule"].apply(make_non_eco)



output_rule_modified.to_csv("output_rules_modified.csv", index=False)