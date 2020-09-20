from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)


text = open('./hobbit.txt').read()
print(text)

# Документ
doc = Doc(text)
print(doc)

# В документе появляются предложения и токены
doc.segment(segmenter)
print(doc.sents[:2])
print(doc.tokens[:2])

# В токенах появляются морфология, синтаксис, факты
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)
print(doc.tokens[:2])
doc.tag_ner(ner_tagger)
print(doc.spans[:2])

# 3 встроенных визуализации: NER, morph и syntax
doc.sents[0].ner.print()
doc.sents[0].morph.print()
doc.sents[0].syntax.print()

# если есть морфология, можно получить леммы
for token in doc.tokens:
    token.lemmatize(morph_vocab)
print({_.text: _.lemma for _ in doc.tokens[:10]})

# Нормализация фактов (обертка над pymorphy)
for span in doc.spans:
    span.normalize(morph_vocab)
print({_.text: _.normal for _ in doc.spans})

# Разбивка фактов на соствляющие
for span in doc.spans:
    if span.type == PER:
        span.extract_fact(names_extractor)
print({_.normal: _.fact.as_dict for _ in doc.spans if _.fact})

# Есть еще встроенные разбивки, основанные на Yargy
# Могут быть не очень точные, нужна настройка
"""
text = '24.01.2017, 2015 год, 2014 г, 1 апреля, май 2017 г., 9 мая 2017 года'
dates = list(dates_extractor(text))
print(dates)
text = '1 599 059, 38 Евро, 420 долларов, 20 млн руб, 20 т. р., 881 913 (Восемьсот восемьдесят одна тысяча девятьсот тринадцать) руб. 98 коп.'
moneys = list(money_extractor(text))
print(moneys)
lines = [
    'Россия, Вологодская обл. г. Череповец, пр.Победы 93 б',
    '692909, РФ, Приморский край, г. Находка, ул. Добролюбова, 18',
    'ул. Народного Ополчения д. 9к.3'
]
for line in lines:
    print(addr_extractor.find(line))
"""
