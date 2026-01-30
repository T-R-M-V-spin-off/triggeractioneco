# Funzione per creare la regola formattata
def format_rule(row):
    # Trasformiamo tutto in minuscolo prima di qualsiasi operazione
    trigger = str(row["triggerTitle"]).lower().strip()
    action = str(row["actionDesc"]).lower().replace("this action will", "then").strip()
    action_short = action.split(".", 1)[0].strip()
    rule_text = f"if {trigger}, {action_short}"
    return rule_text

# Applichiamo la funzione
df_comune["rule"] = df_comune.apply(format_rule, axis=1)

# Mantieni la colonna ID e la colonna rule
df_rules = df_comune[[COLONNA_ID, "rule"]]

# Salva in CSV
df_rules.to_csv("output_rules.csv", index=False)