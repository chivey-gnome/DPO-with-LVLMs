import json
from tqdm import tqdm
import argparse
import os
import base64
from datasets import Dataset

def main(args):
    out = []
    id = 0
    os.mkdir("opa")
    os.mkdir("opa/images")
    for entry in tqdm(os.listdir(args.file)):
        full_path = os.path.join(args.file, entry)
        if not os.path.isfile(full_path):
             continue
        with open(full_path, 'r') as f:
            opadpo = json.load(f)
        for elem in opadpo:
            img = base64.b64decode(elem["image_bytes"])
            s = f"opa/images/{id:08d}.jpg"
            with open(s, "wb") as image_file:
                image_file.write(img)
            id += 1
            try:
                line = {"image": s, "question": elem["query"], "chosen": elem["AI_json_report"]["image description"], "rejected": elem["original_generate_response"]}
            except KeyError:
                line = {"image": s, "question": elem["query"], "chosen": elem["AI_json_report"]["image_description"], "rejected": elem["original_generate_response"]}
            except TypeError:
                 line = {"image": s, "question": elem["query"], "chosen": elem["AI_generate_response"], "rejected": elem["standard_response"]}
            out.append(line)

    dataset = Dataset.from_list(out)
    dataset.save_to_disk("opa/OPA", max_shard_size="10MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to op dpo json")

    args = parser.parse_args()
    main(args)
