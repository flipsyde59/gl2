import moderngl_window as mglw
import moderngl
import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, vector, Matrix33
from PIL import Image

N=100

class Camera():

    def __init__(self, ratio):
        self._field_of_view_degrees = 60.0
        self._z_near = 0.1
        self._z_far = 1000
        self._ratio = ratio
        self.build_projection()

        self._camera_position = Vector3([-20.0, 100.0, -20.0])
        self._camera_front = Vector3([50.0, -50.0, 50.0])
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


class PerspectiveProjection(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 a_position;
                uniform mat4 Mvp;
                uniform mat3 Mn;
                uniform mat4 Mv;
                uniform int N;
                out vec3 v_norm;
                out vec3 v_pos;
                //in vec2 a_textureCoord;
                out vec2 v_textureCoord;
                void main() {
                float x = 50-a_position.x;
			    float z = 50-a_position.y;
		        float y = 50/(1+x*x)+50/(1+z*z);
			    float dx = abs(100*x/((1+x*x)*(1+x*x)));
			    float dz = abs(100*z/((1+z*z)*(1+z*z)));
			    float dy = 1.0;
			    v_color = normalize(Mn*vec3(dx, dy, dz));
		        gl_Position = Mvp * vec4(x, y, z, 1.0);
		        }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_norm;
                in vec3 v_pos;
                out vec4 f_color;
                in vec2 v_textureCoord;
                uniform sampler2D map_first;
                uniform sampler2D map_second;

                void main() {
                const float gamma = 2.2;
                vec3 n = normalize(v_norm);
                vec3 l = normalize(vec3(30.0, 30.0, 3.0) - v_pos);
                vec3 e = normalize(-v_pos);
                float d = max(dot(l, n), 0.1);
                vec3 h = normalize(l + e);
                float s = pow(max(dot(h, n), 0.0), 20.0);
                float temp = 0.5;
                vec4 color = mix(texture(map_first, v_textureCoord),texture(map_second, v_textureCoord), temp);
                vec3 linColor = color.xyz * d + vec3(s);
                f_color = vec4(pow(linColor.x, gamma), pow(linColor.y, gamma), pow(linColor.z, gamma), 1.0);
                }
            ''',
        )

        self.camera = Camera(self.aspect_ratio)
        self.mvp = self.prog['Mvp']
        self.mn = self.prog['Mn']
        self.mv = self.prog['Mv']
        self.prog['map_first'] = 0
        self.prog['map_second'] = 1
        vertices = []
        for i in range(N):
            for j in range(N):
                vertices.append(j)
                vertices.append(i)
        self.vertices1=np.array(vertices)
        self.vbo = self.ctx.buffer(self.vertices1.astype('f4').tobytes()*4)
        render_indicies=[]
        for i in range(N-1):
            a = N * i
            for j in range(N-1):
                render_indicies.append(a)
                render_indicies.append(a+1)
                render_indicies.append(N+a+1)
                render_indicies.append(N+a+1)
                render_indicies.append(N+a)
                render_indicies.append(a)
                a+=1
        render_indicies1=np.array(render_indicies)
        self.index_buffer = self.ctx.buffer(render_indicies1.astype('i4').tobytes()*4)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'a_position', index_buffer=self.index_buffer)
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


    def close(self):
        self.vbo.release()
        self.prog.release()
        self.vao.release()
        self.index_buffer.release()
        self.texture1.release()
        self.texture2.release()
        self.ctx.release()

    def render(self, time, frame_time):
        self.move_camera()

        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.mn.write(Matrix33(self.camera.mat_lookat).inverse.transpose().astype('f4').tobytes())
        self.mv.write(self.camera.mat_lookat.astype('f4').tobytes())
        self.mvp.write((self.camera.mat_projection * self.camera.mat_lookat).astype('f4').tobytes())
        self.prog['N'].value = N
        self.vao.render()



if __name__ == '__main__':
    PerspectiveProjection.run()
