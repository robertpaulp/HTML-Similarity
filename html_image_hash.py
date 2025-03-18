from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from PIL import Image
import imagehash
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ImageAnalyzer:
    @staticmethod
    def get_image_hash(image_path):
        if image_path is None:
            return None
        
        try:
            image = Image.open(image_path)
            hash_value = imagehash.average_hash(image)
            binary = bin(int(str(hash_value), 16))[2:].zfill(64)
            return np.array([int(bit) for bit in binary])

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def hash_clustering(image_hashes):
        valid_hashes = [h for h in image_hashes if h is not None]
        if not valid_hashes:
            return []
        
        hash_array = np.array(valid_hashes)
        n_samples = len(hash_array)
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i,j] = np.sum(hash_array[i] != hash_array[j]) / len(hash_array[i])
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            metric='precomputed',
            linkage='average'
        )
        
        try:
            return clustering.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Clustering error: {str(e)}")
            return []
    
    @staticmethod
    def plot_visual_similarity_matrix(image_hashes, filenames, tier, similarity_matrix):
        if not image_hashes:
            return
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(similarity_matrix), k=1)
        sns.heatmap(similarity_matrix, 
                    xticklabels=[f[:15] for f in filenames],
                    yticklabels=[f[:15] for f in filenames],
                    cmap='YlOrRd',
                    mask=mask,
                    square=True)
        plt.title(f'Visual Similarity Matrix\n{tier}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'hash_plots/{tier}_visual_similarity.png', dpi=300)
        plt.close()
    
    @staticmethod
    def plot_visual_dendrogram(image_hashes, filenames, tier, similarity_matrix):
        """Plot dendrogram for visual clustering"""
        if not image_hashes:
            return
            
        # Distance is 1 - similarity
        distance_matrix = 1 - similarity_matrix
        
        plt.figure(figsize=(12, 8))
        linkage_matrix = linkage(distance_matrix, method='average')
        dendrogram(linkage_matrix, labels=filenames, leaf_rotation=90)
        plt.title(f'Visual Clustering Dendrogram\n{tier}')
        plt.tight_layout()
        plt.savefig(f'hash_plots/{tier}_visual_dendrogram.png', dpi=300)
        plt.close()
        
        return distance_matrix

    @staticmethod
    def process_screenshots(screenshot_dir, tier=None):
        if not os.path.exists(screenshot_dir):
            print(f"Directory not found: {screenshot_dir}")
            return
        
        # Get files from directory
        screenshot_files = sorted([f for f in os.listdir(screenshot_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not screenshot_files:
            print(f"No screenshots found in {screenshot_dir}")
            return
        
        print(f"\nProcessing {len(screenshot_files)} screenshots in {tier}")
        
        image_hashes = []
        valid_files = []
        
        # Process each screenshot
        for filename in screenshot_files:
            path = os.path.join(screenshot_dir, filename)
            hash_value = ImageAnalyzer.get_image_hash(path)
            if hash_value is not None:
                image_hashes.append(hash_value)
                valid_files.append(filename)
            else:
                print(f"Could not generate hash for {filename}")
        
        if not image_hashes:
            print("No valid image hashes generated")
            return
        
        os.makedirs('hash_plots', exist_ok=True)
        
        # Generate visual similarity matrix
        n_samples = len(image_hashes)
        visual_similarity = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                visual_similarity[i,j] = 1 - (np.sum(image_hashes[i] != image_hashes[j]) / len(image_hashes[i]))
        
        ImageAnalyzer.plot_visual_similarity_matrix(image_hashes, valid_files, tier, visual_similarity)
        ImageAnalyzer.plot_visual_dendrogram(image_hashes, valid_files, tier, visual_similarity)


def main():
    print("Image hash clustering")

    for tier in ['tier1', 'tier2', 'tier3', 'tier4']:
        screenshot_dir = f'screenshots/{tier}'
        ImageAnalyzer.process_screenshots(screenshot_dir, tier)


if __name__ == "__main__":
    main()