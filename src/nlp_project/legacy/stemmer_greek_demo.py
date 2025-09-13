try:
    import greek_stemmer  # ή: from greek_stemmer import stemmer as greek_stemmer
except Exception as e:
    print("Skipping Greek stemmer demo: package 'greek_stemmer' not installed:", e)
    raise SystemExit(0)
    
import greek_stemmer
from greek_stemmer import stemmer
input= stemmer.stem_word('εργαζόμενος')
print(input)
