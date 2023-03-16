import torch
def translate_one(text, model, tokenizer, num_beams):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.eval()
    model = model.to(device) 
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = model.generate(**inputs, num_beams=num_beams)
    result = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    return result


