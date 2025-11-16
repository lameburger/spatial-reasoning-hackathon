"""
Beautiful 3D OBJ Mesh & Point Cloud Viewer
Created for viewing ZED camera spatial mapping outputs with hazard detection

Coordinate System: RIGHT_HANDED_Y_UP (matches ZED SDK default)
- X: Right
- Y: Up
- Z: Forward (into the scene)

Controls:
- Left Mouse: Rotate
- Right Mouse / Mouse Wheel: Zoom
- Middle Mouse: Pan
- R: Reset view
- W: Toggle wireframe/solid (mesh only)
- L: Toggle lighting (mesh only)
- H: Toggle hazard points display
- +/-: Increase/Decrease point size (point cloud only)
- ESC: Exit
"""

import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import argparse
import json
from pathlib import Path


class MeshData:
    """Handles OBJ file loading and storage"""
    
    def __init__(self):
        self.vertices = []
        self.colors = []
        self.normals = []
        self.faces = []
        self.face_normals = []
        self.lines = []  # For bounding boxes and other line primitives
        self.bounds_min = None
        self.bounds_max = None
        self.center = None
        self.scale = 1.0
        self.is_point_cloud = False
        
    def load_obj(self, filename):
        """Load OBJ file and compute normals"""
        print(f"Loading mesh from: {filename}")
        
        vertices = []
        colors = []
        normals = []
        faces = []
        lines = []
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        # Vertex - may include RGB color values (ZED SDK format)
                        parts = line.split()
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        # Check if color data is present (ZED SDK adds RGB after xyz)
                        if len(parts) >= 7:
                            colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                    elif line.startswith('vn '):
                        # Normal
                        parts = line.split()
                        normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('l '):
                        # Line element (for bounding boxes)
                        parts = line.split()[1:]
                        line_verts = [int(p) - 1 for p in parts]
                        lines.append(line_verts)
                    elif line.startswith('f '):
                        # Face
                        parts = line.split()[1:]
                        face_vertices = []
                        face_normals = []
                        for part in parts:
                            indices = part.split('/')
                            # Vertex index (subtract 1 for 0-based indexing)
                            face_vertices.append(int(indices[0]) - 1)
                            # Normal index if present
                            if len(indices) > 2 and indices[2]:
                                face_normals.append(int(indices[2]) - 1)
                        faces.append((face_vertices, face_normals))
                        
        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            sys.exit(1)
            
        self.vertices = np.array(vertices, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32) if colors else None
        self.normals = np.array(normals, dtype=np.float32) if normals else None
        self.faces = faces
        self.lines = lines
        
        # Determine if this is a point cloud (no faces)
        self.is_point_cloud = len(self.faces) == 0
        
        if self.is_point_cloud:
            print(f"Loaded: {len(self.vertices)} points (POINT CLOUD)")
        else:
            print(f"Loaded: {len(self.vertices)} vertices, {len(self.faces)} faces (MESH)")
        
        if self.colors is not None:
            print(f"         {len(self.colors)} vertex colors (RGB)")
        
        if self.lines:
            print(f"         {len(self.lines)} line elements (bounding boxes)")
        
        # Compute bounds and center
        self.compute_bounds()
        
        # Compute face normals if vertex normals not provided
        if self.normals is None or len(self.normals) == 0:
            self.compute_normals()
            
    def compute_bounds(self):
        """Compute bounding box and center"""
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)
        self.center = (self.bounds_min + self.bounds_max) / 2.0
        
        # Compute scale to fit in view
        size = np.max(self.bounds_max - self.bounds_min)
        self.scale = 2.0 / size if size > 0 else 1.0
        
        print(f"Mesh bounds: min={self.bounds_min}, max={self.bounds_max}")
        print(f"Center: {self.center}, Scale: {self.scale}")
        
    def compute_normals(self):
        """Compute smooth vertex normals from face normals"""
        print("Computing normals...")
        vertex_normals = np.zeros_like(self.vertices)
        
        for face_verts, _ in self.faces:
            if len(face_verts) >= 3:
                # Get triangle vertices
                v0 = self.vertices[face_verts[0]]
                v1 = self.vertices[face_verts[1]]
                v2 = self.vertices[face_verts[2]]
                
                # Compute face normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                
                # Add to vertex normals (for averaging)
                for idx in face_verts:
                    vertex_normals[idx] += normal
        
        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.normals = vertex_normals / norms
        

class MeshViewer:
    """Beautiful OpenGL mesh viewer with smooth controls"""
    
    def __init__(self, mesh_data, hazards_data=None):
        self.mesh = mesh_data
        self.hazards = hazards_data
        
        # Camera controls (ZED coordinate system: RIGHT_HANDED_Y_UP)
        self.rotation_x = 20.0   # Slight tilt down
        self.rotation_y = 0.0    # Face forward along Z-axis
        self.zoom = 4.0          # Start further back
        self.pan_x = 0.0
        self.pan_y = -0.3        # Slightly lower to center better
        
        # Mouse tracking
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_button = None
        
        # Display options
        self.wireframe = False
        self.lighting = True
        self.point_size = 2.0
        self.show_hazards = True
        
        # Display list for performance
        self.display_list = None
        
    def init_gl(self):
        """Initialize OpenGL settings"""
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Enable back face culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        
        # Key light (bright, from front-top-right)
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 3.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Fill light (softer, from back-left)
        glLightfv(GL_LIGHT1, GL_POSITION, [-2.0, 1.0, -1.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.4, 1.0])
        
        # Material properties - give it a nice appearance
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.25, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.6, 0.65, 0.7, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 32.0)
        
        # Enable color material for easy color changes
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        
        # Set background color (black)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Enable anti-aliasing
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        
        # Build display list for better performance
        self.build_display_list()
        
    def build_display_list(self):
        """Compile mesh/point cloud into a display list for fast rendering"""
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        # Center and scale the mesh/point cloud
        glPushMatrix()
        glScalef(self.mesh.scale, self.mesh.scale, self.mesh.scale)
        glTranslatef(-self.mesh.center[0], -self.mesh.center[1], -self.mesh.center[2])
        
        if self.mesh.is_point_cloud:
            # Render as point cloud
            glPointSize(2.0)  # Nice visible point size
            glBegin(GL_POINTS)
            for i, vertex in enumerate(self.mesh.vertices):
                # Use vertex colors if available
                if self.mesh.colors is not None and i < len(self.mesh.colors):
                    glColor3fv(self.mesh.colors[i])
                else:
                    glColor3f(0.7, 0.75, 0.8)
                glVertex3fv(vertex)
            glEnd()
        else:
            # Render as mesh with faces
            glBegin(GL_TRIANGLES)
            for face_verts, face_normals in self.mesh.faces:
                # Draw triangles (handle quads by triangulation)
                for i in range(len(face_verts) - 2):
                    for j in [0, i + 1, i + 2]:
                        v_idx = face_verts[j]
                        
                        # Set color from vertex colors if available
                        if self.mesh.colors is not None and v_idx < len(self.mesh.colors):
                            glColor3fv(self.mesh.colors[v_idx])
                        
                        # Set normal
                        if face_normals and j < len(face_normals):
                            n_idx = face_normals[j]
                            if n_idx < len(self.mesh.normals):
                                glNormal3fv(self.mesh.normals[n_idx])
                        elif self.mesh.normals is not None and v_idx < len(self.mesh.normals):
                            glNormal3fv(self.mesh.normals[v_idx])
                        
                        # Set vertex
                        glVertex3fv(self.mesh.vertices[v_idx])
            glEnd()
        
        # Render line elements (bounding boxes)
        if self.mesh.lines:
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            glColor3f(1.0, 0.0, 0.0)  # Red for bounding boxes
            glBegin(GL_LINES)
            for line_verts in self.mesh.lines:
                for v_idx in line_verts:
                    if v_idx < len(self.mesh.vertices):
                        glVertex3fv(self.mesh.vertices[v_idx])
            glEnd()
            glLineWidth(1.0)
            if self.lighting:
                glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glEndList()
        
    def display(self):
        """Main display callback"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set up camera
        gluLookAt(0, 0, self.zoom, 0, 0, 0, 0, 1, 0)
        
        # Apply pan
        glTranslatef(self.pan_x, self.pan_y, 0)
        
        # Apply rotations
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Handle display modes
        if self.mesh.is_point_cloud:
            # Point clouds don't use lighting
            glDisable(GL_LIGHTING)
            glPointSize(self.point_size)
        else:
            # Meshes can use wireframe/solid and lighting
            if self.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glDisable(GL_LIGHTING)
                glColor3f(0.7, 0.8, 0.9)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                if self.lighting:
                    glEnable(GL_LIGHTING)
                else:
                    glDisable(GL_LIGHTING)
                glColor3f(0.7, 0.75, 0.8)
        
        # Draw the mesh from display list
        if self.display_list:
            glCallList(self.display_list)
        
        # Draw hazard points if available
        if self.show_hazards and self.hazards:
            self.draw_hazards()
        
        glutSwapBuffers()
        
    def cluster_hazards(self, threshold=0.5):
        """Cluster nearby hazards together for labeling"""
        hazards = self.hazards['hazards']
        if len(hazards) == 0:
            return []
        
        # Get positions
        positions = np.array([h['camera_position'] for h in hazards])
        
        # Simple clustering by distance
        clusters = []
        used = set()
        
        for i, hazard in enumerate(hazards):
            if i in used:
                continue
            
            cluster = {
                'hazards': [hazard],
                'indices': [i],
                'position': np.array(hazard['camera_position'])
            }
            
            # Find nearby hazards
            for j, other_hazard in enumerate(hazards):
                if j <= i or j in used:
                    continue
                
                dist = np.linalg.norm(np.array(hazard['camera_position']) - np.array(other_hazard['camera_position']))
                if dist < threshold:
                    cluster['hazards'].append(other_hazard)
                    cluster['indices'].append(j)
                    used.add(j)
            
            # Calculate mean position
            if len(cluster['hazards']) > 1:
                cluster['position'] = np.mean([np.array(h['camera_position']) for h in cluster['hazards']], axis=0)
            
            clusters.append(cluster)
            used.add(i)
        
        return clusters
    
    def draw_hazards(self):
        """Draw hazard detection points in 3D space with labels"""
        glDisable(GL_LIGHTING)
        glPointSize(15.0)  # Large points for visibility
        
        # Color mapping for hazard classes
        color_map = {
            'person': (1.0, 1.0, 0.0),  # Yellow
            'excavator': (1.0, 0.0, 0.0),  # Red
            'dump_truck': (1.0, 0.5, 0.0),  # Orange
            'no_hardhat': (1.0, 0.0, 0.0),  # Red
            'no_safety_vest': (1.0, 0.0, 0.0),  # Red
            'no_mask': (1.0, 0.5, 0.0),  # Orange
            'safety_cone': (0.0, 1.0, 0.0),  # Green
            'hardhat': (0.0, 1.0, 0.0),  # Green
            'safety_vest': (0.0, 1.0, 0.0),  # Green
            'mask': (0.0, 1.0, 0.0),  # Green
        }
        
        # Cluster hazards for labeling
        clusters = self.cluster_hazards(threshold=0.5)
        
        # Draw points
        glBegin(GL_POINTS)
        for hazard in self.hazards['hazards']:
            pos = hazard['camera_position']
            color = color_map.get(hazard['class'], (1.0, 0.0, 1.0))
            glColor3f(*color)
            
            x = (pos[0] - self.mesh.center[0]) * self.mesh.scale
            y = (pos[1] - self.mesh.center[1]) * self.mesh.scale
            z = (pos[2] - self.mesh.center[2]) * self.mesh.scale
            
            glVertex3f(x, y, z)
        glEnd()
        
        # Draw labels for clusters
        glColor3f(1.0, 1.0, 1.0)  # White text
        for cluster in clusters:
            pos = cluster['position']
            
            # Transform to mesh space
            x = (pos[0] - self.mesh.center[0]) * self.mesh.scale
            y = (pos[1] - self.mesh.center[1]) * self.mesh.scale
            z = (pos[2] - self.mesh.center[2]) * self.mesh.scale
            
            # Get unique classes in this cluster
            classes = set(h['class'] for h in cluster['hazards'])
            
            # Create label - just show class names
            if len(classes) == 1:
                label = list(classes)[0]
            else:
                # Multiple different classes - show all
                label = ", ".join(sorted(classes))
            
            # Draw text label slightly above the point
            self.draw_text_3d(x, y + 0.1, z, label)
        
        glPointSize(1.0)  # Reset point size
    
    def draw_text_3d(self, x, y, z, text):
        """Draw text at 3D position"""
        glRasterPos3f(x, y, z)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
    
    def draw_axes(self):
        """Draw coordinate axes for reference (ZED RIGHT_HANDED_Y_UP system)"""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        axis_length = 0.5
        glBegin(GL_LINES)
        # X axis - Red (Right)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        # Y axis - Green (Up)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        # Z axis - Blue (Forward)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        glEnd()
        
        glLineWidth(1.0)
        if self.lighting:
            glEnable(GL_LIGHTING)
    
    def reshape(self, width, height):
        """Handle window reshape"""
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Match ZED SDK clipping planes: znear=0.5, zfar=100.0
        gluPerspective(45.0, width / height, 0.5, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def mouse(self, button, state, x, y):
        """Handle mouse button events"""
        if state == GLUT_DOWN:
            self.mouse_button = button
            self.mouse_last_x = x
            self.mouse_last_y = y
        else:
            self.mouse_button = None
            
    def motion(self, x, y):
        """Handle mouse motion"""
        dx = x - self.mouse_last_x
        dy = y - self.mouse_last_y
        
        if self.mouse_button == GLUT_LEFT_BUTTON:
            # Rotate
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            
            # Clamp rotation_x to avoid flipping
            self.rotation_x = max(-90, min(90, self.rotation_x))
            
        elif self.mouse_button == GLUT_RIGHT_BUTTON:
            # Zoom
            self.zoom += dy * 0.01
            self.zoom = max(0.5, min(20.0, self.zoom))
            
        elif self.mouse_button == GLUT_MIDDLE_BUTTON:
            # Pan
            self.pan_x += dx * 0.003 * self.zoom
            self.pan_y -= dy * 0.003 * self.zoom
        
        self.mouse_last_x = x
        self.mouse_last_y = y
        glutPostRedisplay()
        
    def mouse_wheel(self, wheel, direction, x, y):
        """Handle mouse wheel for zooming"""
        if direction > 0:
            self.zoom *= 0.9
        else:
            self.zoom *= 1.1
        self.zoom = max(0.5, min(20.0, self.zoom))
        glutPostRedisplay()
        
    def keyboard(self, key, x, y):
        """Handle keyboard events"""
        if key == b'\x1b':  # ESC
            sys.exit(0)
        elif key == b'r' or key == b'R':
            # Reset view
            self.rotation_x = 20.0
            self.rotation_y = 45.0
            self.zoom = 3.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            print("View reset")
        elif key == b'w' or key == b'W':
            # Toggle wireframe (meshes only)
            if not self.mesh.is_point_cloud:
                self.wireframe = not self.wireframe
                print(f"Wireframe: {'ON' if self.wireframe else 'OFF'}")
        elif key == b'l' or key == b'L':
            # Toggle lighting (meshes only)
            if not self.mesh.is_point_cloud:
                self.lighting = not self.lighting
                print(f"Lighting: {'ON' if self.lighting else 'OFF'}")
        elif key == b'+' or key == b'=':
            # Increase point size (point clouds only)
            if self.mesh.is_point_cloud:
                self.point_size = min(10.0, self.point_size + 0.5)
                print(f"Point size: {self.point_size:.1f}")
        elif key == b'-' or key == b'_':
            # Decrease point size (point clouds only)
            if self.mesh.is_point_cloud:
                self.point_size = max(0.5, self.point_size - 0.5)
                print(f"Point size: {self.point_size:.1f}")
        elif key == b'h' or key == b'H':
            # Toggle hazard points display
            if self.hazards:
                self.show_hazards = not self.show_hazards
                print(f"Hazard points: {'ON' if self.show_hazards else 'OFF'}")
            
        glutPostRedisplay()


def main():
    parser = argparse.ArgumentParser(
        description='Beautiful 3D Mesh & Point Cloud Viewer for OBJ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Left Mouse Button    : Rotate view
  Right Mouse Button   : Zoom in/out
  Mouse Wheel          : Zoom in/out
  Middle Mouse Button  : Pan view
  R                    : Reset view
  W                    : Toggle wireframe/solid mode (meshes only)
  L                    : Toggle lighting (meshes only)
  H                    : Toggle hazard points display
  + / -                : Increase/Decrease point size (point clouds only)
  ESC                  : Exit

Example:
  python view_mesh.py mesh_gen.obj
  python view_mesh.py mesh_gen.obj --hazards hazards_detected.json
        """
    )
    parser.add_argument('mesh_file', help='Path to OBJ file to view')
    parser.add_argument('--hazards', type=str, default=None, 
                        help='Path to hazards JSON file (default: hazards_detected.json if exists)')
    args = parser.parse_args()
    
    # Load mesh
    mesh_data = MeshData()
    mesh_data.load_obj(args.mesh_file)
    
    # Load hazards data
    hazards_data = None
    hazards_file = args.hazards
    
    # Auto-detect hazards file if not specified
    if hazards_file is None and Path('hazards_detected.json').exists():
        hazards_file = 'hazards_detected.json'
    
    if hazards_file and Path(hazards_file).exists():
        try:
            with open(hazards_file, 'r') as f:
                hazards_data = json.load(f)
            print(f"Loaded {hazards_data['total_hazards']} hazard detections from: {hazards_file}")
            
            # Count by class
            class_counts = {}
            for hazard in hazards_data['hazards']:
                cls = hazard['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print("Hazard breakdown:")
            for cls, count in sorted(class_counts.items()):
                print(f"  {cls}: {count}")
        except Exception as e:
            print(f"Warning: Could not load hazards file: {e}")
            hazards_data = None
    elif hazards_file:
        print(f"Warning: Hazards file not found: {hazards_file}")
    
    # Initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(1200, 800)
    glutInitWindowPosition(100, 100)
    window_title = f"Mesh Viewer - {args.mesh_file}"
    if hazards_data:
        window_title += f" ({hazards_data['total_hazards']} hazards)"
    glutCreateWindow(window_title.encode())
    
    # Create viewer
    viewer = MeshViewer(mesh_data, hazards_data)
    viewer.init_gl()
    
    # Register callbacks
    glutDisplayFunc(viewer.display)
    glutReshapeFunc(viewer.reshape)
    glutMouseFunc(viewer.mouse)
    glutMotionFunc(viewer.motion)
    glutMouseWheelFunc(viewer.mouse_wheel)
    glutKeyboardFunc(viewer.keyboard)
    
    # Print instructions
    print("\n" + "="*60)
    if mesh_data.is_point_cloud:
        print("POINT CLOUD VIEWER CONTROLS")
    else:
        print("MESH VIEWER CONTROLS")
    print("="*60)
    print("Left Mouse    : Rotate")
    print("Right Mouse   : Zoom")
    print("Mouse Wheel   : Zoom")
    print("Middle Mouse  : Pan")
    print("R             : Reset view")
    if mesh_data.is_point_cloud:
        print("+/-           : Increase/Decrease point size")
    else:
        print("W             : Toggle wireframe/solid")
        print("L             : Toggle lighting")
    print("ESC           : Exit")
    print("="*60 + "\n")
    
    # Start main loop
    glutMainLoop()


if __name__ == "__main__":
    main()

