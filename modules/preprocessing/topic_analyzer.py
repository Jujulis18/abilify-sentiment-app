from .lda_analyzer import LDAAnalyzer
from .bert_analyzer import BERTAnalyzer

class TopicAnalyzer:
	def _init_(self, method='bert'):
		if method == 'lda':
			self.analyzer = LDAAnalyzer()
		else:
			self.analyzer = BERTAnalyzer()

	def analyze(self, texts, **kwargs):
		return self.analyzer.analyze()