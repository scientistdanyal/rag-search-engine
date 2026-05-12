


from cli.lib.keyword_search import tokenize_text
from cli.lib.InvertedIndex import InvertedIndex

my_string = "Danyal is running!"
print(tokenize_text(my_string))

my_index = InvertedIndex()
my_index.__add_document(1, my_string)
my_index.print_index()
my_index.print_docmap()