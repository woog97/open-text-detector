"""Observable surface features, NOT an authorship or prose-quality classifier."""
import re,collections

def signals(text):
 words=re.findall(r'\w+',text.lower());grams=list(zip(*(words[i:] for i in range(8)))) if len(words)>=8 else [];counts=collections.Counter(grams)
 sentences=[' '.join(re.findall(r'\w+',s.lower())) for s in re.split(r'(?<=[.!?])\s+',text)]
 sc=collections.Counter(s for s in sentences if len(s.split())>=8)
 return {'words':len(words),'repeated_8gram_fraction':sum(v-1 for v in counts.values())/max(1,len(grams)), 'repeated_long_sentences':sum(v-1 for v in sc.values()),'code_fence': '```' in text,'assistant_self_reference':bool(re.search(r'\bas an (?:ai|artificial intelligence|language model)\b',text,re.I))}
