def translate_one(text, model, tokenizer):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.eval()
    model = model.to(device) 
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = model.generate(**inputs, num_beams=5)
    predictions_labels_here.append((inputs, tokens))
    result = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    return result