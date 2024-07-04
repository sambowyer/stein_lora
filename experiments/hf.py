from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

text = generator("In this essay, I will discuss", max_length=100, num_return_sequences=5)
print(text)

unmasker = pipeline('fill-mask')
text = unmasker("This is <mask>", top_k=5)
print(text)