import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import os

class ImageAnalyzer:
    def __init__(self):
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def analyze_screenshots(self, screenshots_dir, thresholds, save_dir="clip_plots", threshold=0.5):
        """
        screenshots_dir: Directory containing screenshots
        save_dir: Directory to save plots
        threshold: Distance threshold for clustering
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Analyzing screenshots in {screenshots_dir}")

        image_files = []
        feature_vectors = []
        
        screenshots = sorted(os.listdir(screenshots_dir))
        for filename in screenshots:
            if filename.endswith(".png"):
                image_path = os.path.join(screenshots_dir, filename)
                features = self.extract_features(image_path)
                
                if features is not None:
                    image_files.append(filename)
                    feature_vectors.append(features)
        
        if not feature_vectors:
            print("No images found")
            return
        
        feature_vectors = np.array(feature_vectors)
        similarity_matrix = cosine_similarity(feature_vectors)
        distance_matrix = 1 - similarity_matrix
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='complete'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Visualization
        tier = screenshots_dir.split("/")[-1]
        site_urls = [f.replace(".html.png", "") for f in image_files]

        self._plot_similarity_matrix(similarity_matrix, site_urls, tier, threshold, save_dir)
        self._plot_similarity_dif_threshold(distance_matrix, site_urls, thresholds, tier, save_dir)
        # self._plot_cluster_grid(image_files, labels, screenshots_dir, tier, save_dir)
        
        return similarity_matrix, labels

    def _plot_similarity_matrix(self, similarity_matrix, filenames, tier, threshold, save_dir="clip_plots"):
        plt.figure(figsize=(15, 12))

        sns.heatmap(similarity_matrix,
                   xticklabels=filenames,
                   yticklabels=filenames,
                   cmap='RdYlBu_r',
                   annot=False,)
        plt.title(f'Visual Similarity Matrix\n{tier} threshold={threshold}')
        plt.savefig(f'{save_dir}/{tier}_similarity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cluster_grid(self, image_files, labels, directory, tier, save_dir="clip_plots"):
        """Grid of images grouped by cluster"""
        clusters = {}
        for filename, label in zip(image_files, labels):
            clusters.setdefault(label, []).append(filename)
        
        for label, files in clusters.items():
            n = len(files)
            if n == 0:
                continue
            
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            fig.suptitle(f'Cluster {label} Images\n{tier}')
            
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)
            
            for idx, filename in enumerate(files):
                row = idx // cols
                col = idx % cols
                img = Image.open(os.path.join(directory, filename))
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                axes[row, col].set_title(filename[:20], fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{tier}_cluster_{label}_grid.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_similarity_dif_threshold(self, distance_matrix, filenames, tier, thresholds, save_dir="clip_plots"):
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
        
        plt.title(f'Visual Similarity Dendrogram with Multiple Thresholds\n{tier}')
        plt.xlabel('Screenshots')
        plt.ylabel('Distance')
        plt.legend(title='Thresholds')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{tier}_threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    analyzer = ImageAnalyzer()

    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
    
    screenshot_dir = "screenshots"
    for tier in ['tier1', 'tier2', 'tier3', 'tier4']:
        if os.path.exists(f"{screenshot_dir}/{tier}"):
            analyzer.analyze_screenshots(f"{screenshot_dir}/{tier}", thresholds, save_dir="clip_plots", threshold=0.2)
        else:
            print(f"Directory not found: {screenshot_dir}/{tier}")

if __name__ == "__main__":
    main()
