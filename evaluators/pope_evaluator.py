from .evaluator import Evaluator
import os
import json
import torch
from PIL import Image
from tqdm import tqdm

class PopeEvaluator(Evaluator):
    def __init__(self, args):
        self.pope_path=args.pope_path
        self.coco_path=args.coco_path
        self.set_name=args.set_name
        self.type_name=args.model_type
        self.output_dir=args.output_dir

    def eval(self, model, processor):
        """
        Run inference on the POPE benchmark dataset.
        """
        # Determine dataset file
        if self.set_name == "random":
            questions_file = os.path.join(self.pope_path, "coco_pope_random.json")
        elif self.set_name == "popular":
            questions_file = os.path.join(self.pope_path, "coco_pope_popular.json")
        elif self.set_name == "adv":
            questions_file = os.path.join(self.pope_path, "coco_pope_adversarial.json")
        else:
            raise ValueError(f"Unknown set: {self.set_name}")
        
        answer_file = f"pope_{self.set_name}_{self.type_name}.jsonl"

        # Load questions
        with open(questions_file, "r") as f:
            lines = f.readlines()

        results = []
        print("==============================================================")
        for line in tqdm(lines, desc=f"Generating on {self.set_name}"):
            data = json.loads(line)
            question = data["text"]
            image_name = data["image"]
            question_id = data["question_id"]
            label = data["label"]

            # Load image
            image_path = os.path.join(self.coco_path, "val2014", image_name)
            image = Image.open(image_path).convert("RGB")

            # Prepare prompt
            chat_data = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            prompt = processor.apply_chat_template(chat_data, add_generation_prompt=True)

            # Process inputs
            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to("cuda")

            # Generate answer
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            results.append({
                "question_id": question_id,
                "question": question,
                "answer": output_text,
                "label": label
            })

        # Save results
        os.makedirs(self.output_dir, exist_ok=True)
        out_file = os.path.join(self.output_dir, answer_file)
        with open(out_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        print(f"Saved predictions to {out_file}")
        print("Benchmark: POPE done.")
        # Could do this in memory, but use artifacts on disk for now
        return (questions_file, out_file)