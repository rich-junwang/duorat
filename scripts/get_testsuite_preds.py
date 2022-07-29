import sys
import json
import re

duorat_output_file = sys.argv[1]
original_data_file = sys.argv[2]
gold_file = sys.argv[3]
gold_fixed_file = sys.argv[4]
testsuite_output_file = sys.argv[5]

ignore_patterns = None
if len(sys.argv) > 6:
    ignore_patterns = sys.argv[6]
print(ignore_patterns)

fout_testsuite = open(testsuite_output_file, 'w')
fout_gold_fixed = open(gold_fixed_file, "w")

gold_predictions = []
with open(original_data_file, "r") as f, open(gold_file) as f_gold:
    original_data = json.load(f)

    for line in f_gold:
        line = line.strip()
        if line:
            gold_predictions.append(line)

    i = 0
    fixed_gold_predictions = []
    for example in original_data:
        interaction = example["interaction"]

        new_interaction = []
        for utter_info in interaction:
            if ignore_patterns and re.search(ignore_patterns, utter_info["utterance"]):
                print(f"ignored: {utter_info}")
                i += 1
                continue
            new_interaction.append(utter_info)
            fixed_gold_predictions.append(gold_predictions[i])
            i += 1
        example["interaction"] = new_interaction
    gold_predictions = fixed_gold_predictions

predictions = []
count_empty_preds = 0
count_preds = 0
with open(duorat_output_file, "r") as f:
    prev_inferred_code = ""
    for line in f:
        line = line.strip()
        entry_data = json.loads(line)
        if len(entry_data["beams"]) > 0:
            predictions.append(entry_data["beams"][0]["inferred_code"])
            prev_inferred_code = predictions[-1]
        else:
            predictions.append(prev_inferred_code)
            count_empty_preds += 1
        count_preds += 1
print(f"There are {count_empty_preds} out of {count_preds} predictions with empty output(s).")

assert len(gold_predictions) == len(predictions)

i = 0
for example in original_data:
    interaction = example["interaction"]

    for pred, gold_pred in zip(predictions[i:i+len(interaction)],
                               gold_predictions[i:i+len(interaction)]):
        fout_testsuite.write(f"{pred}\n")
        fout_gold_fixed.write(f"{gold_pred}\n")
        i += 1
    fout_testsuite.write("\n")
    fout_gold_fixed.write("\n")

print(f"No. of actual predictions: {i}")

fout_testsuite.flush()
fout_testsuite.close()
fout_gold_fixed.flush()
fout_gold_fixed.close()
