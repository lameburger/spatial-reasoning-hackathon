#!/usr/bin/env python3
"""
Simple HTTP server to view 3D mesh and hazards in browser.
Run this script and open http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
import subprocess
import json
from pathlib import Path

PORT = 8000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests - serve static files"""
        # Serve static files
        super().do_GET()

    def do_POST(self):
        """Handle POST requests for triggering spatial.py"""
        if self.path == '/run_scan':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    post_data = self.rfile.read(content_length)
            except:
                pass
            
            try:
                # Check if spatial.py exists
                if not Path('spatial.py').exists():
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'spatial.py not found'}).encode())
                    return
                
                # Start spatial.py in a new window
                print("\n[SERVER] Launching spatial.py...")
                
                if os.name == 'nt':  # Windows
                    # Use start command to open in new window
                    subprocess.Popen(['start', 'cmd', '/k', 'python', 'spatial.py'], shell=True)
                else:  # Linux/Mac
                    subprocess.Popen(['python', 'spatial.py'])
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'Scan launched', 'message': 'spatial.py window opened'}).encode())
                
            except Exception as e:
                print(f"[ERROR] Failed to start scan: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())

    def log_message(self, format, *args):
        # Log errors and POST requests
        if '404' in str(args) or '500' in str(args):
            print(f"[ERROR] {args[0]}")
        elif 'POST' in str(args):
            print(f"[REQUEST] {args[0]}")

def main():
    # Check if required files exist
    if not Path('web_viewer.html').exists():
        print("Error: web_viewer.html not found!")
        return
    
    print("\n" + "="*60)
    print("3D Mesh & Hazard Viewer Server")
    print("="*60)
    print(f"\nServer URL: http://localhost:{PORT}/web_viewer.html")
    print("\nRequired files:")
    print(f"  - mesh_gen.obj: {'✓ Found' if Path('mesh_gen.obj').exists() else '✗ Missing (run spatial.py to generate)'}")
    print(f"  - hazards_detected.json: {'✓ Found' if Path('hazards_detected.json').exists() else '✗ Missing (run spatial.py with hazard detection)'}")
    
    if not Path('mesh_gen.obj').exists() or not Path('hazards_detected.json').exists():
        print("\nNote: Run spatial.py first to generate the required files.")
    
    print("\nViewer Controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("  - Click hazard point: Show details and VLM analysis")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Open browser automatically
    webbrowser.open(f'http://localhost:{PORT}/web_viewer.html')
    
    # Start server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        try:
            print(f"Server running... (visit http://localhost:{PORT}/web_viewer.html)\n")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            httpd.shutdown()

if __name__ == "__main__":
    main()

