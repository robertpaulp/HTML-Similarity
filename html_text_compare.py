from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class HTMLProcessor:
    """Processes HTML documents for text analysis"""
    
    @staticmethod
    def extract_visible_text(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)
    
    @staticmethod
    def load_documents(directory):
        documents = {}
        for filename in os.listdir(directory):
            if filename.endswith('.html'):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    documents[filename] = HTMLProcessor.extract_visible_text(file.read())
        return documents

class ClusterAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def compute_features(self, documents):
        return self.vectorizer.fit_transform(documents.values())
    
    def cluster_documents(self, tfidf_matrix, threshold=0.7):
        similarity_matrix = cosine_similarity(tfidf_matrix)
        distance_matrix = 1 - similarity_matrix
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='average'
        )
        
        return clustering.fit_predict(distance_matrix), similarity_matrix, distance_matrix
        
    def plot_similarity_matrix(self, similarity_matrix, filenames, tier, threshold, save_dir="text_based_plots"):
        plt.figure(figsize=(15, 12))

        sns.heatmap(similarity_matrix,
                   xticklabels=filenames,
                   yticklabels=filenames,
                   cmap='RdYlBu_r',
                   annot=False,)
        plt.title(f'Text Similarity Matrix\n{tier} threshold={threshold}')
        plt.savefig(f'{save_dir}/{tier}_similarity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dendrogram_dif_thresholds(self, distance_matrix, filenames, tier, thresholds, save_dir="text_based_plots"):
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        plt.figure(figsize=(20, 15))
        linkage_matrix = linkage(distance_matrix, method='complete')

        dendrogram(
            linkage_matrix,
            labels=filenames,
        )
        
        for threshold, color in zip(thresholds, colors):
            plt.axhline(y=threshold, color=color, linestyle='--', 
                    label=f'Threshold = {threshold}')
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='precomputed',
                linkage='complete'
            )
            labels = clustering.fit_predict(distance_matrix)
            n_clusters = len(set(labels))
            
            plt.text(len(filenames) + 1, threshold + 0.01, 
                    f'{n_clusters} clusters', 
                    color=color, fontweight='bold')
        
        plt.title(f'Text Similarity Dendrogram with Multiple Thresholds\n{tier}')
        plt.xlabel('Screenshots')
        plt.ylabel('Distance')
        plt.legend(title='Thresholds')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{tier}_threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("Text-based clustering")
    html_processor = HTMLProcessor()
    cluster_analyzer = ClusterAnalyzer()
    
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
    threshold = 0.5
    
    for tier in ['tier1', 'tier2', 'tier3', 'tier4']:
        print(f"Processing tier: {tier}")
        directory = f"clones/{tier}/"
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        
        docs = html_processor.load_documents(directory)
        if not docs:
            continue
        
        features = cluster_analyzer.compute_features(docs)
        
        for threshold in thresholds:
            _, similarity_matrix, distance_matrix = cluster_analyzer.cluster_documents(features, threshold)

        cluster_analyzer.plot_similarity_matrix(similarity_matrix, list(docs.keys()), tier, threshold)
        cluster_analyzer.plot_dendrogram_dif_thresholds(distance_matrix, list(docs.keys()), tier, thresholds)

# def print_results(tier, threshold, metrics, docs, labels):
#     print('--------------------------------')
#     print(f'Tier: {tier}')
#     print('--------------------------------')
#     print(f"\nResults for threshold = {threshold}:")
#     print(f"Number of clusters: {metrics['n_clusters']}")
#     print(f"Average similarity within clusters: {metrics['avg_similarity_within_clusters']:.3f}")
#     print(f"Silhouette score: {metrics['silhouette_score']}")
#     print('--------------------------------')
    
#     groups = {}
#     for filename, label in zip(docs.keys(), labels):
#         groups.setdefault(label, []).append(filename)
    
#     print("\nClusters:")
#     for label, files in groups.items():
#         print(f"\nCluster {label}:")
#         for file in files:
#             print(f"  - {file}")

if __name__ == "__main__":
    main()
