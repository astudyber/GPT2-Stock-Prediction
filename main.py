# 导入需要使用的模型和 tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 假设您的自定义模型和 tokenizer 可以与 GPT2LMHeadModel 和 GPT2Tokenizer 兼容，则加载模型和 tokenizer。
tokenizer = GPT2Tokenizer.from_pretrained('./model/')
model = GPT2LMHeadModel.from_pretrained('./model/')

# 设置输入文本
text = "I love your"

# 对输入文本进行编码
input_ids = tokenizer.encode(text, return_tensors='pt')

# 生成唯一文本
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=1, temperature=1.0, top_k=50, pad_token_id=tokenizer.eos_token_id)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)


#-------------------------------------------------------------
# # 使用束搜索方法生成多个文本序列
# num_sequences = 3  # 指定要生成的文本序列数
# output = model.generate(input_ids, max_length=20, num_return_sequences=num_sequences, temperature=1.0, top_k=50, pad_token_id=tokenizer.eos_token_id, num_beams=5)
#
# # 解码并输出每个生成的文本序列
# for i in range(num_sequences):
#     decoded_output = tokenizer.decode(output[i], skip_special_tokens=True)
#     print(f"Generated sequence {i+1}: {decoded_output}")
# -------------------------------------------------------------
