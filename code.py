import moderngl_window as mglw
import moderngl
import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, vector

class Camera():

    def __init__(self, ratio):
        self._field_of_view_degrees = 60.0
        self._z_near = 0.1
        self._z_far = 100
        self._ratio = ratio
        self.build_projection()

        self._camera_position = Vector3([0.0, 0.0, -40.0])
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

    def strafe_up(self,y=0.1):
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


def grid(size, steps):
    u = np.repeat(np.linspace(-size, size, steps), 2)
    v = np.tile([-size, size], steps)
    w = np.zeros(steps * 2)
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


class PerspectiveProjection(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_vert;
                void main() {
                    gl_Position = Mvp * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 f_color;
                void main() {
                    f_color = vec4(0.1, 0.1, 0.1, 1.0);
                }
            ''',
        )

        self.camera = Camera(self.aspect_ratio)
        self.mvp = self.prog['Mvp']
        self.vbo = self.ctx.buffer(grid(15, 100).astype('f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')
        self.states = {
            self.wnd.keys.W: False,     # forward
            self.wnd.keys.S: False,     # backwards
            self.wnd.keys.UP: False,    # strafe Up
            self.wnd.keys.DOWN: False,  # strafe Down
            self.wnd.keys.A: False,     # strafe left
            self.wnd.keys.D: False,     # strafe right
            self.wnd.keys.Q: False,     # rotate left
            self.wnd.keys.E: False,     # rotare right
            self.wnd.keys.Z: False,     # rotare up
            self.wnd.keys.X: False,     # rotare down
            self.wnd.mouse.left: False,
        }

    #def mouse_position_event(self, x, y, dx, dy):
        #print("Mouse position:", x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        if dy<0: self.camera.rotate_up(-dy/10)
        if dy>0: self.camera.rotate_down(dy/10)
        if dx>0: self.camera.rotate_right(dx/10)
        if dx<0: self.camera.rotate_left(-dx/10)


    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        if y_offset>0: self.camera.zoom_in(y_offset)
        elif y_offset<0: self.camera.zoom_out(-y_offset)


    #def mouse_press_event(self, x, y, button):
        #print("Mouse button {} pressed at {}, {}".format(button, x, y))

    #def mouse_release_event(self, x: int, y: int, button: int):
        #print("Mouse button {} released at {}, {}".format(button, x, y))

    def move_camera(self):
        if self.states.get(self.wnd.keys.W):
            self.camera.move_forward()

        if self.states.get(self.wnd.keys.S):
            self.camera.move_backwards()

        if self.states.get(self.wnd.keys.UP):
            self.camera.strafe_up()

        if self.states.get(self.wnd.keys.DOWN):
            self.camera.strafe_down()

        if self.states.get(self.wnd.keys.A):
            self.camera.strafe_left()

        if self.states.get(self.wnd.keys.D):
            self.camera.strafe_right()

        if self.states.get(self.wnd.keys.Q):
            self.camera.rotate_left()

        if self.states.get(self.wnd.keys.E):
            self.camera.rotate_right()

        if self.states.get(self.wnd.keys.Z):
            self.camera.rotate_up()

        if self.states.get(self.wnd.keys.X):
            self.camera.rotate_down()

        if self.states.get(self.wnd.mouse.left):
            self.camera.move_forward()


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
        self.vao.render(moderngl.LINES)


if __name__ == '__main__':
    PerspectiveProjection.run()
