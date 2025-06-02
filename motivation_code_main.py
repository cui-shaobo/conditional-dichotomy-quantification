"""
@Project  : dichotomous-score
@File     : motivation_code_main.py
@Author   : Shaobo Cui
@Date     : 04.09.2024 17:01
"""
from oppositescore.model import AnglE
from oppositescore.utils import cosine_similarity

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
# for non-retrieval tasks, we don't need to specify prompt when using UAE-Large-V1.
z = 'A company launches a revolutionary product. The company gains a significant market share. '
doc_vecs = angle.encode([
    z + "The product's unique features attract a quite large customer base.",
    z + 'Competitors quickly release similar products, reducing the company’s advantage.',
    z + 'People frequently share their happy usage experience on social media. '
])

for i, dv1 in enumerate(doc_vecs):
    print(i)
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
