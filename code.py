import moderngl_window as mglw
import moderngl
import numpy as np
from pyrr import Matrix33, Matrix44, Quaternion, Vector3, vector

N=100

class Camera():

    def __init__(self, ratio):
        self._field_of_view_degrees = 60.0
        self._z_near = 0.1
        self._z_far = 100
        self._ratio = ratio
        self.build_projection()

        self._camera_position = Vector3([50.0, 50.0, -70.0])
        self._camera_front = Vector3([0.0, 0.0, 1.0])
        self._camera_up = Vector3([0.0, 1.0, 0.0])
        self._cameras_target = (self._camera_position + self._camera_front)
        self.build_look_at()

    def zoom_in(self, z=0.1):
        self._field_of_view_degrees = self._field_of_view_degrees - z
        self.build_projection()

    def zoom_out(self, z=0.1):
        self._field_of_view_degrees = self._field_of_view_degrees + z
        self.build_projection()

    def move_forward(self, step=0.1):
        self._camera_position = self._camera_position + self._camera_front * step
        self.build_look_at()

    def move_backwards(self, step=0.1):
        self._camera_position = self._camera_position - self._camera_front * step
        self.build_look_at()

    def strafe_left(self, x=0.1):
        self._camera_position = self._camera_position - vector.normalize(self._camera_front ^ self._camera_up) * x
        self.build_look_at()

    def strafe_right(self, x=0.1):
        self._camera_position = self._camera_position + vector.normalize(self._camera_front ^ self._camera_up) * x
        self.build_look_at()

    def strafe_up(self, y=0.1):
        self._camera_position = self._camera_position + self._camera_up * y
        self.build_look_at()

    def strafe_down(self, y=0.1):
        self._camera_position = self._camera_position - self._camera_up * y
        self.build_look_at()

    def rotate_left(self, x=0.1):
        rotation = Quaternion.from_y_rotation(2 * float(x) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_right(self, x=0.1):
        rotation = Quaternion.from_y_rotation(-2 * float(x) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_up(self, y=0.1):
        rotation = Quaternion.from_x_rotation(2 * float(y) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_down(self, y=0.1):
        rotation = Quaternion.from_x_rotation(-2 * float(y) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def build_look_at(self):
        self._cameras_target = (self._camera_position + self._camera_front)
        self.mat_lookat = Matrix44.look_at(
            self._camera_position,
            self._cameras_target,
            self._camera_up)

    def build_projection(self):
        self.mat_projection = Matrix44.perspective_projection(
            self._field_of_view_degrees,
            self._ratio,
            self._z_near,
            self._z_far)


def grid(size):                             #generator of net
    matrix = np.zeros(size * size * 2)
    for i in range(size):
        for j in range(size):
            tmp = 2 * (i * size + j)
            matrix[tmp] = j                 #x
            matrix[tmp + 1] = i             #z
    return matrix

def indexes(size):
    matrix = np.zeros((size-1)*(size-1)*6)
    for i in range(size-1):
        for j in range(size-1):
            tmp = 6 * (i * (size - 1) + j)
            current = i * size + j
            next_ = (i + 1) * size + j
            matrix[tmp] = current
            matrix[tmp + 1] = current + 1
            matrix[tmp + 2] = next_ + 1
            matrix[tmp + 3] = next_ + 1
            matrix[tmp + 4] = next_
            matrix[tmp + 5] = current
    return matrix

class PerspectiveProjection(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location = 0) in vec2 a_position;
                uniform mat4 Mvp;
                uniform mat3 Mn;
                out vec3 v_color;
                void main() {
                float x = a_position.x;
			    float z = a_position.y;
		        float y = -cos(-1-(x*x+z*z))-exp(0.2-(x*x+z*z));
			    float dx = abs(-2*exp(0.2 - x*x - z*z)*x + 2*x*sin(1 + x*x + z*z));
			    float dz = abs(-2*exp(0.2 - x*x - z*z)*z + 2*z*sin(1 + x*x + z*z));
			    float dy = 1.0;
			    v_color = normalize(Mn * vec3(dx, dy, dz));
		        gl_Position = Mvp * vec4(x, y, z, 1.0);
		        }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            ''',
        )

        self.camera = Camera(self.aspect_ratio)
        self.mvp = self.prog['Mvp']
        self.mn = self.prog['Mn']
        self.vbo = self.ctx.buffer(grid(N).astype('f4').tobytes())
        self.ibo = self.ctx.buffer(indexes(N).astype('f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'a_position')

        self.states = {
            self.wnd.keys.W: False,  # forward
            self.wnd.keys.S: False,  # backwards
            self.wnd.keys.UP: False,  # strafe Up
            self.wnd.keys.DOWN: False,  # strafe Down
            self.wnd.keys.LEFT: False,  # strafe left
            self.wnd.keys.RIGHT: False,  # strafe right
            self.wnd.keys.Q: False,     #zoom out
            self.wnd.keys.E: False,     #zoom in
            self.wnd.keys.A: False,     # rotate left
            self.wnd.keys.D: False,  # rotare right
            self.wnd.keys.Z: False,  # rotare up
            self.wnd.keys.X: False,  # rotare down
            self.wnd.mouse.left: False, #drag left mouse=rotate
            self.wnd.mouse.right: False, #drag right mouse=strafe
        }
        #self.prog.release()
        #self.vbo.release()
        #self.vao.release()

    # def mouse_position_event(self, x, y, dx, dy):
    # print("Mouse position:", x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        if self.wnd._mouse_buttons.left == True:
            if dy < 0: self.camera.rotate_up(-dy / 10)
            if dy > 0: self.camera.rotate_down(dy / 10)
            if dx > 0: self.camera.rotate_right(-dx / 10)
            if dx < 0: self.camera.rotate_left(dx / 10)
        if self.wnd._mouse_buttons.right == True:
            if dy < 0: self.camera.strafe_up(dy / 10)
            if dy > 0: self.camera.strafe_down(-dy / 10)
            if dx > 0: self.camera.strafe_right(-dx / 10)
            if dx < 0: self.camera.strafe_left(dx / 10)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        if y_offset > 0:
            self.camera.move_forward(y_offset)
        elif y_offset < 0:
            self.camera.move_backwards(-y_offset)

    #def mouse_press_event(self, x, y, button):
        #print("Mouse button {} pressed at {}, {}".format(button, x, y))

    # def mouse_release_event(self, x: int, y: int, button: int):
    # print("Mouse button {} released at {}, {}".format(button, x, y))

    def move_camera(self):
        if self.states.get(self.wnd.keys.W):
            self.camera.move_forward()

        if self.states.get(self.wnd.keys.S):
            self.camera.move_backwards()

        if self.states.get(self.wnd.keys.UP):
            self.camera.strafe_up()

        if self.states.get(self.wnd.keys.DOWN):
            self.camera.strafe_down()

        if self.states.get(self.wnd.keys.LEFT):
            self.camera.strafe_left()

        if self.states.get(self.wnd.keys.RIGHT):
            self.camera.strafe_right()

        if self.states.get(self.wnd.keys.A):
            self.camera.rotate_left()

        if self.states.get(self.wnd.keys.D):
            self.camera.rotate_right()

        if self.states.get(self.wnd.keys.Z):
            self.camera.rotate_up()

        if self.states.get(self.wnd.keys.X):
            self.camera.rotate_down()

        if self.states.get(self.wnd.keys.Q):
            self.camera.zoom_out()

        if self.states.get(self.wnd.keys.E):
            self.camera.zoom_in()


    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS or action == self.wnd.mouse:
            self.states[key] = True
        else:
            self.states[key] = False

    def render(self, time, frame_time):
        self.move_camera()

        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.mvp.write((self.camera.mat_projection * self.camera.mat_lookat).astype('f4').tobytes())
        self.mn.write(np.linalg.inv(Matrix33(self.camera.mat_lookat)).transpose().astype('f4').tobytes())
        self.vao.render()#moderngl.LINES



if __name__ == '__main__':
    PerspectiveProjection.run()
