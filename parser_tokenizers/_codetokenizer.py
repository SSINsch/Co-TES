# CodeBERT/GraphCodeBERT/translation/parser/utils.py 참조
from tree_sitter import Language, Parser
import logging

logger = logging.getLogger(__name__)


class CodeTokenizer:
    def __init__(self, lang='c'):
        """
        :param lang: [c, cpp, c_sharp] 가능
        """
        self.lang = lang
        self.target_lang = Language('./parser_tokenizers/my-languages.so', self.lang)
        self.parser = Parser()
        self.parser.set_language(self.target_lang)

    def tree_to_token_index(self, root_node):
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            return [(root_node.start_point, root_node.end_point, root_node.type)]
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens += self.tree_to_token_index(child)
            return code_tokens

    @staticmethod
    def index_to_code_token(index, code):
        start_point = index[0]
        end_point = index[1]
        if start_point[0] == end_point[0]:
            s = code[start_point[0]][start_point[1]:end_point[1]]
        else:
            s = ""
            s += code[start_point[0]][start_point[1]:]
            for i in range(start_point[0] + 1, end_point[0]):
                s += code[i]
            s += code[end_point[0]][:end_point[1]]
        return s

    def tokenize(self, code):
        bytes_code = bytes(code, 'utf8')
        tree = self.parser.parse(bytes_code)
        root_node = tree.root_node
        code = code.split('\n')
        tokens_index = self.tree_to_token_index(root_node)
        code_tokens = [self.index_to_code_token(x, code) for x in tokens_index]

        return code_tokens
