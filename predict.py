# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List

from cog import BaseModel, BasePredictor, File, Input
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

# os.environ["TRANSFORMERS_VERBOSITY"] = "info"


MODEL_NAME = "openai/clip-vit-large-patch14"
CACHE_DIR = ".transformer"


class Output(BaseModel):
    embedding: List[float]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.model: CLIPModel = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)  # type: ignore
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    def predict(
        self,
        text: str = Input(description="Input text", default=None),
        image: File = Input(description="Input image", default=None),
    ) -> Output:
        """Run a single prediction on the model"""

        if image:
            image = Image.open(image)

            inputs = self.processor(images=image, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            embedding = image_features.tolist()[0]
        elif text:
            inputs = self.tokenizer([text], padding=True, return_tensors="pt")
            text_features = self.model.get_text_features(**inputs)  # type: ignore
            embedding = text_features.tolist()[0]
        else:
            raise Exception("Missing inputs.")

        return Output(embedding=embedding)
