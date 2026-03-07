## Diário

### 2025-03-05

* Foi definido que o material de base para treinamento do modelo será o arquivo "TCC versão 4.6.pdf"

### 2025-03-06

* Testes nas bibliotecas PyMuPDF e PyPDF. A primeira deu um resultado muito mais satisfatório.

Código de exemplo:

```py
doc = pymupdf.open("TCC  versao 4.6.pdf") # open a document
_out = open("TCC  versao 4.6.txt", "wb") # create a text output
for _page in doc: # iterate the document pages
    _text = _page.get_text().encode("utf8") # get plain text (is in UTF-8)
    _out.write(text) # write text of page
    _out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
_out.close()
```

* Pacotes demonstrados

```py
from sentence_transformers impor SentenceTransformer
import chromadb
import hashlib
```

* Modelos: https://huggingface.co/PORTULAN