import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import os

from html_text_compare import HTMLProcessor
from html_visual_analyzer import ImageAnalyzer as ClipImageAnalyzer
from html_image_hash import ImageHashAnalyzer
from html_text_compare import ClusterAnalyzer

class EnsembleClusterer:
    def __init__(self):
        self.text_processor = HTMLProcessor()
        self.text_analyzer = ClusterAnalyzer()
        self.hash_analyzer = ImageHashAnalyzer()
        self.clip_analyzer = ClipImageAnalyzer()

    def ensemble_clustering(self, tier):
        text_labels, text_files, text_similarity = self.get_text_clusters(f"clones/{tier}")
        print("Finished text clustering")
        print("--------------------------------")
        
        hash_labels, hash_files, hash_similarity = self.get_hash_clusters(f"screenshots/{tier}")
        print("Finished hash clustering")
        print("--------------------------------")
        
        clip_labels, clip_files, clip_similarity = self.get_clip_clusters(f"screenshots/{tier}")
        print("Finished clip clustering")
        print("--------------------------------")
        
        if not all([text_labels is not None, hash_labels is not None, clip_labels is not None]):
            print(f"Could not get clustering results for {tier}")
            return None, None, None
        
        all_files = sorted(set(text_files + hash_files + clip_files))
        file_to_idx = {f: i for i, f in enumerate(all_files)}

        text_similarity = self.align_similarity_matrix(text_files, text_similarity, all_files, file_to_idx)
        hash_similarity = self.align_similarity_matrix(hash_files, hash_similarity, all_files, file_to_idx)
        clip_similarity = self.align_similarity_matrix(clip_files, clip_similarity, all_files, file_to_idx)
        
        if text_similarity is None or hash_similarity is None or clip_similarity is None:
            print("Similarity matrices have inconsistent dimensions")
            return None, None, None

        combined_similarity = self.combine_similarity_matrices(text_similarity, hash_similarity, clip_similarity)
        
        distance_matrix = 1 - combined_similarity
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric='precomputed', linkage='complete')
        
        final_labels = clustering.fit_predict(distance_matrix)
        
        return final_labels, all_files, combined_similarity

    def align_similarity_matrix(self, files, similarity_matrix, all_files, file_to_idx):
        n_files = len(all_files)
        aligned_similarity = np.zeros((n_files, n_files))
        
        for i, file1 in enumerate(files):
            for j, file2 in enumerate(files):
                idx1 = file_to_idx.get(file1)
                idx2 = file_to_idx.get(file2)
                
                if idx1 is not None and idx2 is not None:
                    aligned_similarity[idx1, idx2] = similarity_matrix[i, j]
                    aligned_similarity[idx2, idx1] = similarity_matrix[i, j]
        
        return aligned_similarity


    def combine_similarity_matrices(self, text_similarity, hash_similarity, clip_similarity):
        weights = {
            'text': 0.1,
            'hash': 0.2,
            'clip': 0.7
        }
        
        text_similarity = text_similarity / np.max(text_similarity)
        hash_similarity = hash_similarity / np.max(hash_similarity)
        clip_similarity = clip_similarity / np.max(clip_similarity)

        if text_similarity.shape != hash_similarity.shape or hash_similarity.shape != clip_similarity.shape:
            print("All similarity matrices must have the same dimensions")
            return None
        
        combined_similarity = weights['text'] * text_similarity + weights['hash'] * hash_similarity + weights['clip'] * clip_similarity

        return combined_similarity

    def plot_combined_similarity(self, similarity_matrix, filenames, tier):
        plt.figure(figsize=(15, 12))
        sns.heatmap(similarity_matrix, xticklabels=filenames, yticklabels=filenames, cmap='RdYlBu_r', annot=False)
        plt.title(f'Combined Similarity Matrix\n{tier}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        os.makedirs('ensemble_plots', exist_ok=True)
        plt.savefig(f'ensemble_plots/{tier}_combined_similarity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def get_text_clusters(self, directory):
        docs = self.text_processor.load_documents(directory)
        if not docs:
            return None, None, None
            
        features = self.text_analyzer.compute_features(docs)
        labels, similarity_matrix, distance_matrix = self.text_analyzer.cluster_documents(features, threshold=0.5)
        return labels, list(docs.keys()), similarity_matrix
    
    def get_hash_clusters(self, screenshot_dir):
        if not os.path.exists(screenshot_dir):
            return None, None, None
            
        screenshot_files = sorted([f for f in os.listdir(screenshot_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        image_hashes = []
        valid_files = []
        
        for filename in screenshot_files:
            path = os.path.join(screenshot_dir, filename)
            hash_value = self.hash_analyzer.get_image_hash(path)
            if hash_value is not None:
                image_hashes.append(hash_value)
                valid_files.append(filename)
        
        if not image_hashes:
            return None, None, None
            
        hash_array = np.array(image_hashes)
        n_samples = len(hash_array)
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                similarity = 1 - (np.sum(hash_array[i] != hash_array[j]) / len(hash_array[i]))
                similarity_matrix[i,j] = similarity
        
        labels = self.hash_analyzer.hash_clustering(image_hashes)
        return labels, valid_files, similarity_matrix
    
    def get_clip_clusters(self, screenshot_dir):
        if not os.path.exists(screenshot_dir):
            return None, None, None
            
        similarity_matrix, labels, site_urls = self.clip_analyzer.analyze_screenshots(
            screenshot_dir, 
            save_dir="clip_plots",
            threshold=0.2
        )
        
        if labels is None:
            return None, None, None
            
        image_files = sorted([f for f in os.listdir(screenshot_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
        return labels, image_files, similarity_matrix

def main():
    print("Ensemble clustering")
    print("--------------------------------")
    ensemble = EnsembleClusterer()
    
    with open('clustering_results.txt', 'w') as f:
        for tier in ['tier1', 'tier2', 'tier3', 'tier4']:
            print(f"\nProcessing {tier}")
            labels, files, combined_similarity = ensemble.ensemble_clustering(tier)
            
            if labels is not None:
                normalized_files = {}
                for filename in files:
                    base_name = filename.replace('.html.html', '.html')
                    normalized_files[filename] = base_name
                
                clusters = {}
                for filename, label in zip(files, labels):
                    normalized_name = normalized_files[filename]
                    clusters.setdefault(label, set()).add(normalized_name)
                
                f.write(f"\nTier {tier[-1]}:\n")
                cluster_groups = []
                
                for label, cluster_files in clusters.items():
                    cluster_groups.append(sorted(list(cluster_files)))
                
                cluster_groups.sort(key=lambda x: (-len(x), x[0]))
                
                for group in cluster_groups:
                    if len(group) > 1:
                        f.write(f"[{', '.join(group)}]\n")
                    
                for group in cluster_groups:
                    if len(group) == 1:
                        f.write(f"[{group[0]}]\n")
            
            f.write("\n")
    
    print("\nResults have been saved to clustering_results.txt")

if __name__ == "__main__":
    main()
