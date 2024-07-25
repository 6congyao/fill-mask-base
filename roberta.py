from transformers import pipeline
unmasker = pipeline("fill-mask", model="FacebookAI/roberta-base")


def unmask(sentence):
    return unmasker(sentence)[0]


demo_inputs = [
    "Furthermore, <mask> population growth in many developing countries puts a strain on natural resources and infrastructure.",
    "As populations grow, so does the demand for energy, water, and food, frequently leading to overexploitation and <mask> degradation.",
    "This <mask> cycle of resource depletion and environmental degradation contributes to rising temperatures and the vulnerability of these regions."
]

results = [unmask(s) for s in demo_inputs]

# print(unmasker(text))
for prediction in results:
    print(f"Score: {prediction['score']}, Token: {prediction['token_str']}")
