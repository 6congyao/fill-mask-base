from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
model = AutoModel.from_pretrained("xwjzds/detime", trust_remote_code=True)
model.eval()
# outputs = []
text = """
The phenomenon of rising temperatures is a pressing global issue, with profound implications for the Third World, often referred to as developing countries.""" #make sure to add Repeat at the beginning

# inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length = 512)
inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length = 512).input_ids
attention = tokenizer(text, return_tensors="pt", padding='max_length', max_length = 512).attention_mask
# inputs_id = inputs.input_ids.to(model.device)
# attention = inputs.attention_mask.to(model.device)
outputs = model.generate(inputs, attention, max_length = 512)
# output = model.model.encoder(inputs_id, attention).last_hidden_state   #batch size * seq length * embedding size, 
# output = model.encoder(output)
# outputs.append(output.detach().cpu())

print(tokenizer.decode(outputs[0]))
#Now decoder_output will output low quality text generation
