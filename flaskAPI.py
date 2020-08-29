from flask import Flask
from flask import request
from flask_cors import CORS

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address



import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

application = Flask(__name__)
CORS(application)
limiter = Limiter(
    application,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

model = T5ForConditionalGeneration.from_pretrained('t5_paraphrase_msrp')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

max_len = 256



# API definition
@application.route("/paraphrase-api", methods=['POST'])
@limiter.limit("2000 per day")
def translation():
    if request.json:
        sentence = request.json["text"]
        text =  "paraphrase: " + sentence + " </s>"
        print(sentence)
        encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=5
        )
        output = ""
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sentence != "":
                output = sent
        print(output)
        return output

if __name__ == '__main__':
    application.run(debug=False, threaded=True)


