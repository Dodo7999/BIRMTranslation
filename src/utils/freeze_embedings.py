

def freeze_emb(model, type):
    if type == "mt5":
        for param in model.shared.parameters():
            param.requires_grad = False
        for param in model.encoder.embed_tokens.parameters():
            param.requires_grad = False
    else:
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False
    return model