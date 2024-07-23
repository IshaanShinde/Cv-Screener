# text_analyzer.py

import json
import torch
import spacy
import gensim
import numpy as np

from sklearn import feature_extraction, metrics
from transformers import AutoTokenizer, AutoModel

class TextAnalyzer:
    def __init__(self):
        self.nlp = self.load_spacy_model()
        self.word_embeddings = self.load_word_embeddings()
        self.bert_model, self.bert_tokenizer = self.load_bert_model()
        self.skills_ontology = self.load_skills_ontology()
        self.tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

    def extract_skills(self, text):
        preprocessed_text = self.preprocess_text(text)
        return [skill for skill in self.skills_ontology if skill in preprocessed_text]

    def compute_tf_idf(self, text):
        return self.tfidf_vectorizer.fit_transform([text]).toarray()[0]

    def compute_word_embeddings(self, text):
        words = self.preprocess_text(text).split()
        word_vectors = [self.word_embeddings[word] for word in words if word in self.word_embeddings]
        if not word_vectors:
            return np.zeros(self.word_embeddings.vector_size)
        return np.mean(word_vectors, axis=0)

    def compute_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def compute_similarity(self, text1, text2):
        embedding1 = self.compute_bert_embedding(text1)
        embedding2 = self.compute_bert_embedding(text2)
        return metrics.pairwise.cosine_similarity([embedding1], [embedding2])[0][0]

    def rank_documents(self, query, documents):
        query_embedding = self.compute_bert_embedding(query)
        doc_embeddings = [self.compute_bert_embedding(doc) for doc in documents]
        similarities = metrics.pairwise.cosine_similarity([query_embedding], doc_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        return [(documents[i], similarities[i]) for i in ranked_indices]

    def classify_document(self, text):
        # This is a placeholder. In a real implementation, you'd use a trained classifier.
        embedding = self.compute_bert_embedding(text)
        # Assume we have predefined category embeddings
        categories = ["Software Development", "Data Science", "Project Management"]
        category_embeddings = [self.compute_bert_embedding(cat) for cat in categories]
        similarities = metrics.pairwise.cosine_similarity([embedding], category_embeddings)[0]
        return list(zip(categories, similarities))

    @staticmethod
    def load_spacy_model():
        return spacy.load("en_core_web_sm")

    @staticmethod
    def load_word_embeddings():
        # In a real implementation, you'd load pre-trained embeddings
        return gensim.models.KeyedVectors.load_word2vec_format('path_to_embeddings', binary=True)

    @staticmethod
    def load_bert_model():
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        return model, tokenizer

    @staticmethod
    def load_skills_ontology():
        # In a real implementation, you'd load this from a file or database
        return ["python", "java", "machine learning", "data analysis", "project management"]

class RecruitmentAnalyzer(TextAnalyzer):
    def __init__(self):
        super().__init__()

    def analyze_job_description(self, description):
        preprocessed_desc = self.preprocess_text(description)
        skills = self.extract_skills(preprocessed_desc)
        embedding = self.compute_bert_embedding(preprocessed_desc)
        categories = self.classify_document(preprocessed_desc)
        return {
            "skills": skills,
            "embedding": embedding.tolist(),
            "categories": categories
        }

    def analyze_cv(self, cv_text):
        preprocessed_cv = self.preprocess_text(cv_text)
        skills = self.extract_skills(preprocessed_cv)
        embedding = self.compute_bert_embedding(preprocessed_cv)
        categories = self.classify_document(preprocessed_cv)
        return {
            "skills": skills,
            "embedding": embedding.tolist(),
            "categories": categories
        }

    def match_cv_to_job(self, cv_text, job_description):
        cv_analysis = self.analyze_cv(cv_text)
        job_analysis = self.analyze_job_description(job_description)
        
        skill_match = len(set(cv_analysis["skills"]) & set(job_analysis["skills"]))
        embedding_similarity = metrics.pairwise.cosine_similarity(
            [cv_analysis["embedding"]], [job_analysis["embedding"]]
        )[0][0]
        
        category_similarity = sum(
            cv_cat[1] * job_cat[1] 
            for cv_cat, job_cat in zip(cv_analysis["categories"], job_analysis["categories"])
        ) / len(cv_analysis["categories"])
        
        overall_score = (skill_match + embedding_similarity + category_similarity) / 3
        
        return {
            "overall_score": overall_score,
            "skill_match": skill_match,
            "semantic_similarity": embedding_similarity,
            "category_match": category_similarity
        }

if __name__ == "__main__":
    analyzer = RecruitmentAnalyzer()
    job_desc = "We are looking for a Python developer with machine learning experience."
    cv = "Experienced software engineer with 5 years of Python and data analysis experience."
    
    job_requirements = analyzer.analyze_job_description(job_desc)
    cv_info = analyzer.analyze_cv(cv)
    match_result = analyzer.match_cv_to_job(cv, job_desc)
    
    print(json.dumps(match_result, indent=2))