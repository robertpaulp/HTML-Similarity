from scipy.cluster.hierarchy import dendrogram, linkage
from selenium import webdriver
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import socket
import time
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

class LocalServer:
    """Handles local server for serving HTML files"""
    
    @staticmethod
    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            return s.getsockname()[1]
    
    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.port = self.get_free_port()
        self.server = None
    
    def start(self):
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self_, *args, **kwargs):
                super().__init__(*args, directory=self.directory, **kwargs)
        
        self.server = HTTPServer(('localhost', self.port), Handler)
        thread = threading.Thread(target=self.server.serve_forever)
        thread.daemon = True
        thread.start()
    
    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

class ScreenshotCapture:
    """Handles webpage screenshot capture"""
    
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--window-size=1920,1080')
        self.options.add_argument('--page-load-timeout=30')
        self.options.add_argument('--script-timeout=30')
    
    def capture(self, filename, directory, port):
        driver = None
        try:
            driver = webdriver.Chrome(options=self.options)
            
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(30)
            
            url = f"http://localhost:{port}/{os.path.basename(directory)}/{filename}"
            print(f"Accessing: {url}")
            driver.get(url)
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(2)
            
            tier = os.path.basename(directory)
            os.makedirs('screenshots', exist_ok=True)
            os.makedirs(f'screenshots/{tier}', exist_ok=True)
            screenshot_path = f'screenshots/{tier}/{filename}.png'
            
            driver.save_screenshot(screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            print(f"Error capturing screenshot for {filename}: {str(e)}")
            return None
        finally:
            if driver:
                driver.quit()

def main():
    print("Capturing screenshots")

    screenshot_capture = ScreenshotCapture()
    server = LocalServer("clones")
    server.start()

    time.sleep(1)

    try:        
        for tier in ['tier1', 'tier2', 'tier3', 'tier4']:
            directory = os.path.join("clones", tier)
            if not os.path.exists(directory):
                continue

            print(f"Working on {directory}")
            for filename in os.listdir(directory):
                if filename.endswith('.html'):
                    screenshot_capture.capture(filename, directory, server.port)

    finally:
        server.stop()

if __name__ == "__main__":
    main()
