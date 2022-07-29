import json
import argparse


def get_preds_from_json_file(preds_json_file: str,
                             gold_txt_file: str,
                             output_preds_txt_file: str) -> None:
    with open(preds_json_file) as inpf, open(gold_txt_file) as inpf_gold, open(output_preds_txt_file, 'w') as outf:
        preds = []
        for line, line_gold in zip(inpf, inpf_gold):
            line = line.strip()
            db_id = line_gold.strip().split('\t')[1].strip()
            preds.append((json.loads(line), db_id))

        for pred in preds:
            query = pred[0]["beams"][0]["inferred_code"]
            outf.write(f"{query}\t{pred[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-json-file", required=True)
    parser.add_argument("--gold-txt-file", required=True)
    parser.add_argument("--output-preds-txt-file", required=True)
    args = parser.parse_args()

    get_preds_from_json_file(preds_json_file=args.preds_json_file,
                             gold_txt_file=args.gold_txt_file,
                             output_preds_txt_file=args.output_preds_txt_file)
