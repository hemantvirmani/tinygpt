

txt = "word suggestions — Got the wrong word? Simply press backspace or click the word to reveal a dropdown with alternative suggestions. शब्द सुझाव — गलत शब्द मिला? वैकल्पिक सुझावों के साथ ड्रॉपडाउन प्रकट करने के लिए बस बैकस्पेस दबाएं या शब्द पर क्लिक करें।"

utf8_tokens = txt.encode("utf-8")
utf8_tokens = list(map(int, utf8_tokens))

print("---")
print("length of txt: ", len(txt))
#print(txt)
print("---")
print("length of tokens: ", len(utf8_tokens))
#print(utf8_tokens)
print("---")

def get_stats(ids):
    counter = {}
    for token_pair in zip(ids, ids[1:]):
        counter[token_pair] = counter.get(token_pair, 0) + 1
    return counter

token_stats = get_stats(utf8_tokens)

print("length of token stats: ", len(token_stats))
#print(token_stats)

#print(sorted(((v, k) for k,v in token_stats.items()), reverse=False))
#print('\n'.join(str(item) for item in sorted(((v, k) for k,v in token_stats.items()), reverse=True)))

