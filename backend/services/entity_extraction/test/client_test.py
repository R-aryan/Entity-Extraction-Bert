import joblib
import torch

# from backend.common.logging.console_loger import ConsoleLogger
# from backend.services.entity_extraction.application.ai.inference.prediction_manager import PredictionManager
# from backend.services.entity_extraction.application.ai.training.src.preprocess import Preprocess
from backend.services.entity_extraction.application.ai.settings import Settings
from backend.services.entity_extraction.application.ai.model import BERTEntityModel
from backend.services.entity_extraction.application.ai.training.src.dataset import BERTEntityDataset

mappings = joblib.load(Settings.MAPPING_PATH)
enc_pos = mappings["enc_pos"]
enc_tag = mappings["enc_tag"]

num_pos = len(enc_pos)
num_tag = len(enc_tag)

sample_sentence = "Ritesh is coming to india"

tokenized_sentence = Settings.TOKENIZER.encode(sample_sentence)
tokenized_str = Settings.TOKENIZER.tokenize(sample_sentence)
sample_sentence = sample_sentence.split()
print(sample_sentence)
print(tokenized_sentence)
print(tokenized_str)

inverse_enc_pos = {}
inverse_enc_tag = {}

for k, v in enc_pos.items():
    inverse_enc_pos[v] = k

for k, v in enc_tag.items():
    inverse_enc_tag[v] = k

print(enc_pos)
print("-" * 70)
print(enc_tag)

print(inverse_enc_pos)
print("-" * 70)
print(inverse_enc_tag)

test_dataset = BERTEntityDataset(
    texts=[sample_sentence],
    pos=[[0] * len(sample_sentence)],
    tags=[[0] * len(sample_sentence)]
)

model = BERTEntityModel(num_tag=num_tag, num_pos=num_pos)
model.load_state_dict(torch.load(Settings.WEIGHTS_PATH, map_location=torch.device('cpu')))
model.to(Settings.DEVICE)

device = Settings.DEVICE
model.eval()
with torch.no_grad():
    data = test_dataset[0]
    b_input_ids = data['input_ids']
    b_attention_mask = data['attention_mask']
    b_token_type_ids = data['token_type_ids']
    b_target_pos = data['target_pos']
    b_target_tag = data['target_tag']

    # moving tensors to device
    b_input_ids = b_input_ids.to(device).unsqueeze(0)
    b_attention_mask = b_attention_mask.to(device).unsqueeze(0)
    b_token_type_ids = b_token_type_ids.to(device).unsqueeze(0)
    b_target_pos = b_target_pos.to(device).unsqueeze(0)
    b_target_tag = b_target_tag.to(device).unsqueeze(0)

    tag, pos, _ = model(
        input_ids=b_input_ids,
        attention_mask=b_attention_mask,
        token_type_ids=b_token_type_ids,
        target_pos=b_target_pos,
        target_tag=b_target_tag
    )

    tags = tag.argmax(2).cpu().numpy().reshape(-1)
    pos = pos.argmax(2).cpu().numpy().reshape(-1)
    print(tags)
    print(pos)

    l_pos = []
    l_tag = []
    for i in range(len(tokenized_sentence)):
        if i == 0 or i == len(tokenized_sentence) - 1:
            continue
        l_pos.append(inverse_enc_pos[pos[i]])
        l_tag.append(inverse_enc_tag[tags[i]])

    print("Original Sentence -- ", sample_sentence)
    print("Tokenized Sentence -- ", tokenized_sentence)
    print("POS  -- ", (l_pos))
    print("TAG--- ", str(l_tag))

# p1 = PredictionManager(preprocess=Preprocess(), logger=ConsoleLogger(filename=Settings.LOGS_DIRECTORY))
#
# print("Sample Input, ", sentence)
# output = p1.run_inference(sentence)
# print(output)
