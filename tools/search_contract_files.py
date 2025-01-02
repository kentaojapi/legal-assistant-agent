import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sudachipy import dictionary, tokenizer


@dataclass
class SearchResult:
    file_name: str
    score: float
    matched_text: str
    context: str
    full_text: str


@dataclass
class Document:
    file_name: str
    full_text: str


class DocumentSearcher:
    """
    ドキュメント検索クラス
        NOTE: 簡易的なクラスのため効率的実装ではない
    """
    DIRECTORY = "./documents/markdown/"

    def __init__(self) -> None:
        """
        Initialize the searcher with documents directory
        """
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        self.documents: list[Document] = []
        self._load_documents(self.DIRECTORY)

    def _load_documents(self, directory: str) -> None:
        for filename in os.listdir(directory):
            if filename.endswith('.md'):
                path = os.path.join(directory, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    document = Document(file_name=filename, full_text=f.read())
                    self.documents.append(document)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize Japanese text into words using Sudachi
        """
        tokens = self.tokenizer_obj.tokenize(text, self.mode)
        words = [token.dictionary_form() for token in tokens 
                if token.part_of_speech()[0] in ['名詞', '動詞', '形容詞']]
        return words

    def _get_context(self, text: str, query: str, window_size: int = 30) -> str:
        """
        Get context around the matched query
        """
        index = text.find(query)
        if index == -1:
            return ""
        
        start = max(0, index - window_size)
        end = min(len(text), index + len(query) + window_size)
        
        context = '...' + text[start:end] if start > 0 else text[start:end]
            
        if end < len(text):
            context += "..."
            
        return context

    def search(self, query: str, threshold: float = 0.3) -> List[SearchResult]:
        """
        Search for query in all documents
        
        Args:
            query: Search query in Japanese
            threshold: Minimum score threshold for results
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        query_tokens = set(self._tokenize(query))
        results = []

        for document in self.documents:
            # 完全一致検索
            if query in document.full_text:
                context = self._get_context(document.full_text, query)
                results.append(SearchResult(
                    file_name=document.file_name,
                    score=1.0,
                    matched_text=query,
                    context=context
                ))
                continue

            # トークンベースの部分一致検索
            doc_tokens = set(self._tokenize(document.full_text))
            matching_tokens = query_tokens & doc_tokens
            
            if matching_tokens:
                score = len(matching_tokens) / len(query_tokens)
                if score >= threshold:
                    # 最も関連性の高い部分を抽出してコンテキストとする
                    matched_text = "、".join(matching_tokens)
                    # マッチした最初のトークンの周辺をコンテキストとして抽出
                    first_token = list(matching_tokens)[0]
                    context = self._get_context(document.full_text, first_token)
                    
                    results.append(SearchResult(
                        file_name=document.file_name,
                        score=score,
                        matched_text=matched_text,
                        context=context,
                        full_text=document.full_text
                    ))
        # スコアで降順ソート
        return sorted(results, key=lambda x: x.score, reverse=True)[0]

def main():
    # 使用例
    searcher = DocumentSearcher()
    
    # 検索例
    query = "契約 書類 作成"
    results = searcher.search(query)
    
    print(f"検索クエリ: {query}\n")
    for result in results:
        print(f"ドキュメント: {result.file_name}")
        print(f"スコア: {result.score:.2f}")
        print(f"マッチしたテキスト: {result.matched_text}")
        print(f"コンテキスト: {result.context}")
        print("-" * 50)

if __name__ == "__main__":
    main()