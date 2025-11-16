"""
OpenGL Viewer with ZED SDK Object Rendering
Properly renders sl.Objects with 3D bounding boxes and labels
"""

from ogl_viewer.viewer import *
import pyzed.sl as sl
from OpenGL.GLUT import *


class GLViewer(GLViewer):
    """Extended viewer with object rendering support"""
    
    def __init__(self):
        super().__init__()
        self.objects = sl.Objects()  # Store ZED Objects
        self.draw_objects = True
        
    def update_objects(self, objects):
        """Update the objects to render"""
        self.mutex.acquire()
        self.objects = objects
        self.mutex.release()
    
    def draw_callback(self):
        """Main draw callback with object rendering"""
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0, 0, 0, 1.0)

            self.mutex.acquire()
            self.update()
            
            # Draw image
            if self.available:
                self.image_handler.draw()

            # Draw mesh if tracking OK
            if self.tracking_state == sl.POSITIONAL_TRACKING_STATE.OK and len(self.sub_maps) > 0:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                tmp = self.pose
                tmp.inverse()
                proj = (self.projection * tmp).m
                vpMat = proj.flatten()
                
                glUseProgram(self.shader_image.get_program_id())
                glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE, vpMat)
                glUniform3fv(self.shader_color_loc, 1, self.vertices_color)
        
                for m in range(len(self.sub_maps)):
                    self.sub_maps[m].draw(self.draw_mesh)

                glUseProgram(0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
            # Draw 3D bounding boxes from ZED Objects
            if self.draw_objects and self.tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                self.render_3d_bounding_boxes()
            
            self.print_text()
            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()
    
    def render_3d_bounding_boxes(self):
        """
        Render 3D bounding boxes from ZED SDK Objects
        The bounding_box field contains 8 3D points in world coordinates
        """
        if not self.objects.is_new or len(self.objects.object_list) == 0:
            return
        
        # Setup transformation matrices
        glUseProgram(0)  # Use fixed-function pipeline
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        proj_array = np.array(self.projection.m).reshape(4, 4).T.flatten()
        glMultMatrixf(proj_array)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        view_mat = np.array(self.pose.m).reshape(4, 4)
        view_mat_inv = np.linalg.inv(view_mat)
        glMultMatrixf(view_mat_inv.T.flatten())
        
        # Drawing settings
        glLineWidth(3.0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Draw each object's 3D bounding box
        for obj in self.objects.object_list:
            if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                continue
            
            # Get color based on object class
            color = self.get_object_color(obj.label)
            glColor3f(color[0], color[1], color[2])
            
            # ZED SDK provides 8 corners of 3D bounding box
            # Order: [0-3] = top face corners, [4-7] = bottom face corners
            bbox = obj.bounding_box
            
            if len(bbox) != 8:
                continue
            
            # Draw bounding box edges
            glBegin(GL_LINES)
            
            # Top face (corners 0-3)
            glVertex3f(bbox[0][0], bbox[0][1], bbox[0][2])
            glVertex3f(bbox[1][0], bbox[1][1], bbox[1][2])
            
            glVertex3f(bbox[1][0], bbox[1][1], bbox[1][2])
            glVertex3f(bbox[2][0], bbox[2][1], bbox[2][2])
            
            glVertex3f(bbox[2][0], bbox[2][1], bbox[2][2])
            glVertex3f(bbox[3][0], bbox[3][1], bbox[3][2])
            
            glVertex3f(bbox[3][0], bbox[3][1], bbox[3][2])
            glVertex3f(bbox[0][0], bbox[0][1], bbox[0][2])
            
            # Bottom face (corners 4-7)
            glVertex3f(bbox[4][0], bbox[4][1], bbox[4][2])
            glVertex3f(bbox[5][0], bbox[5][1], bbox[5][2])
            
            glVertex3f(bbox[5][0], bbox[5][1], bbox[5][2])
            glVertex3f(bbox[6][0], bbox[6][1], bbox[6][2])
            
            glVertex3f(bbox[6][0], bbox[6][1], bbox[6][2])
            glVertex3f(bbox[7][0], bbox[7][1], bbox[7][2])
            
            glVertex3f(bbox[7][0], bbox[7][1], bbox[7][2])
            glVertex3f(bbox[4][0], bbox[4][1], bbox[4][2])
            
            # Vertical edges connecting top and bottom
            glVertex3f(bbox[0][0], bbox[0][1], bbox[0][2])
            glVertex3f(bbox[4][0], bbox[4][1], bbox[4][2])
            
            glVertex3f(bbox[1][0], bbox[1][1], bbox[1][2])
            glVertex3f(bbox[5][0], bbox[5][1], bbox[5][2])
            
            glVertex3f(bbox[2][0], bbox[2][1], bbox[2][2])
            glVertex3f(bbox[6][0], bbox[6][1], bbox[6][2])
            
            glVertex3f(bbox[3][0], bbox[3][1], bbox[3][2])
            glVertex3f(bbox[7][0], bbox[7][1], bbox[7][2])
            
            glEnd()
            
            # Draw object label at top of bounding box
            self.draw_3d_label(obj)
        
        # Restore state
        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_3d_label(self, obj):
        """Draw text label at object position"""
        # Get object name from label enum
        label_name = self.get_object_label_name(obj.label)
        
        # Position text slightly above the object's center
        pos = obj.position
        text_pos = [pos[0], pos[1] + 0.3, pos[2]]  # 30cm above center
        
        # Get color for this object
        color = self.get_object_color(obj.label)
        
        # Draw the text
        glColor3f(color[0], color[1], color[2])
        glRasterPos3f(text_pos[0], text_pos[1], text_pos[2])
        
        # Format label with ID and confidence
        label_text = f"{label_name} #{obj.id} ({obj.confidence:.0%})"
        
        # Render each character
        for char in label_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
    
    def get_object_label_name(self, label):
        """Convert ZED object label enum to readable string"""
        label_names = {
            sl.OBJECT_CLASS.PERSON: "PERSON",
            sl.OBJECT_CLASS.VEHICLE: "VEHICLE",
            sl.OBJECT_CLASS.BAG: "BAG",
            sl.OBJECT_CLASS.ANIMAL: "ANIMAL",
            sl.OBJECT_CLASS.ELECTRONICS: "ELECTRONICS",
            sl.OBJECT_CLASS.FRUIT_VEGETABLE: "FRUIT/VEG",
            sl.OBJECT_CLASS.SPORT: "SPORTS",
        }
        
        return label_names.get(label, str(label).split('.')[-1])
    
    def get_object_color(self, label):
        """Get color for object class"""
        # Color mapping for common ZED object classes
        color_map = {
            sl.OBJECT_CLASS.PERSON: [0.0, 1.0, 0.0],      # Green
            sl.OBJECT_CLASS.VEHICLE: [1.0, 0.0, 0.0],     # Red
            sl.OBJECT_CLASS.BAG: [0.0, 0.0, 1.0],         # Blue
            sl.OBJECT_CLASS.ANIMAL: [1.0, 1.0, 0.0],      # Yellow
            sl.OBJECT_CLASS.ELECTRONICS: [1.0, 0.0, 1.0], # Magenta
            sl.OBJECT_CLASS.FRUIT_VEGETABLE: [0.0, 1.0, 1.0], # Cyan
        }
        
        return color_map.get(label, [1.0, 1.0, 1.0])  # White default

