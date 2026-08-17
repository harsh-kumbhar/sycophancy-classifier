import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_PATH = "model/final_model"


class SycophancyModel:
    """
    Loads and manages the trained DeBERTa
    sycophancy classification model.
    """

    def __init__(self):
        self.device = torch.device("cpu")

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        print("Loading model...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

        print("✓ Sycophancy model loaded")
        print("Device:", self.device)
        print("Parameters:", self.model.num_parameters())

    def predict(self, conversation):
        """
        Run inference on a complete conversation.
        """

        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predicted_id = torch.argmax(
            probabilities
        ).item()

        prediction = self.id2label[
            predicted_id
        ]

        confidence = probabilities[
            predicted_id
        ].item()

        probability_dict = {
            self.id2label[i]:
            probabilities[i].item()
            for i in range(len(probabilities))
        }

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probability_dict,
        }