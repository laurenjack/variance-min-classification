from typing import List

class HistoricalClick:
    pass

class CandidateArticle:
    pass

class RankedArticle:
    pass


class ArticleRanker:

    def rank(self, articles: List[CandidateArticle], clicks: List[HistoricalClick]) -> List[RankedArticle]:
        pass
