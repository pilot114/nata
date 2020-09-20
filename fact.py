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


text = open('./potter.txt').read()

# Документ
doc = Doc(text)

# В документе появляются предложения и токены
doc.segment(segmenter)

# В токенах появляются морфология, синтаксис, факты
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)
doc.tag_ner(ner_tagger)

# Нормализация фактов (обертка над pymorphy)
facts = []
for span in doc.spans:
    span.normalize(morph_vocab)
    if span.type == PER:
        facts.append(span.normal)
uniq = set(facts)
print(uniq)
