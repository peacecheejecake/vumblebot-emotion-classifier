import argparse
import torch
from transformers import BertTokenizerFast

from emotions import emotion_abstract_list, emotion_specific_list
from model import load_model


class Run:

    def __init__(self, config):
        plm_name = f"kykim/{'al' if config.albert else ''}bert-kor-base"
        self.tokenizer = BertTokenizerFast.from_pretrained(plm_name)
        self.model = load_model(config)
        self.emotion_decoder = self.create_emotion_decoder()

    def create_emotion_decoder(self):
        all_emotions = []
        for first, seconds in zip(emotion_abstract_list, emotion_specific_list):
            all_emotions.extend(
                f"{first}-{second}" for second in seconds)
        return {i: e for i, e in enumerate(all_emotions)}

    @torch.no_grad()
    def classify(self, input):
        inputs = self.tokenizer(
            input,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        output = self.model(inputs).argmax(-1).cpu().item()
        return self.emotion_decoder[output]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i")
    parser.add_argument("--model_path", "-m",
                        default="./assets/tmpn_albert.pth")
    parser.add_argument("--albert", "-a", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=False)

    args = parser.parse_args()

    run = Run(args)

    print(run.classify(args.input))
