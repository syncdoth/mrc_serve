from abc import ABC
import json
import logging
import os

from fairseq import hub_utils
from fairseq.models.roberta import RobertaHubInterface, RobertaModel
import mecab
from pororo.models.brainbert.BrainRoBERTa import BrainRobertaHubInterface
from pororo.tasks.machine_reading_comprehension import PororoBertMrc
from pororo.tasks.utils.base import TaskConfig
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer
from pororo.utils import postprocess_span

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

DATA_PATH = "absolute path to the mrc_model folder"

class MRCHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        logger.debug('Will load from {0}'.format(model_dir))
        # Read model serialize/pt file
        x = hub_utils.from_pretrained(
            model_dir,
            "model.pt",
            DATA_PATH,
            load_checkpoint_heads=True
        )
        model_interface = BrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            model_dir,
        ).to(self.device)

        tagger = mecab.MeCab()

        self.model = PororoBertMrc(model_interface, tagger, postprocess_span, TaskConfig("mrc", "ko", "brainbert.base.ko.korquad"))



        # Read the mapping file, index to object name
        # mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        # if os.path.isfile(mapping_file_path):
        #     with open(mapping_file_path) as f:
        #         self.mapping = json.load(f)
        # else:
        #     logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes.
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        question, text = sentences.split("|")
        logger.info("Received question: '%s'", question)

        return {"question": question, "text": text}

    def inference(self, input):
        """
        Predict the class of a text using a trained transformer model.
        """
        prediction = self.model(input["question"], input["text"])
        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = MRCHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
