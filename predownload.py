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

ckpt_dir = download_or_load("bert/brainbert.base.ko.korquad", "ko")
tok_path = download_or_load(f"tokenizers/bpe32k.ko.zip", "ko")

# ckpt_dir = "/root/.pororo/bert/brainbert.base.ko.korquad"
# tok_path = "/root/.pororo/tokenizers/bpe32k.ko"


x = hub_utils.from_pretrained(
    ckpt_dir,
    "model.pt",
    ckpt_dir,
    load_checkpoint_heads=True
)
model = BrainRobertaHubInterface(
    x["args"],
    x["task"],
    x["models"][0],
    tok_path,
).to(torch.device("cuda"))

tagger = mecab.MeCab()
final = PororoBertMrc(model, tagger, postprocess_span, TaskConfig("mrc", "ko", "brainbert.base.ko.korquad"))

print(final("이름이 뭐야?", "이름은 시리야."))