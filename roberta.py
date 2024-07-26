from transformers import pipeline
unmasker = pipeline("fill-mask", model="FacebookAI/roberta-base")


def unmask(sentence):
    return unmasker(sentence)


demo_inputs = [
    "Furthermore, <mask> population growth in many developing countries puts a strain on natural resources and infrastructure.",
    "<mask> Digital devices and online websites are capturing the interests of younger generations as never before and parents are facing more issues when it comes to raising their kids, a lot of which is they're doing through technology.",
    "This <mask> cycle of resource depletion and environmental degradation contributes to rising temperatures and the vulnerability of these regions."
]

# results = [unmask(s) for s in demo_inputs]

# for prediction in results:
#     print(f"Score: {prediction['score']}, Token: {prediction['token_str']}")
