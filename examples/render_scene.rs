use raytracing_tutorial::geometry::{Sphere, SE3, Plane};
use raytracing_tutorial::matrix::Matrix33;
use raytracing_tutorial::vector::{V3};
use raytracing_tutorial::scene::{Camera, Scene, Element, Material, SceneGeometry,
     SurfaceType, Color, Light, DirectionalLight, PointLight, Coloration, Texture};
use image::RgbImage;
use raytracing_tutorial::rendering::{render_depth,render_image};

#[allow(non_snake_case)]
fn main() {
    //create camera
    let eye = V3::new([0.0, -5.0, 2.0]);
    let target = V3::new([0.0, 0.0, 1.0]);
    let up = V3::new([0.0, 0.0, 1.0]);
    let pose = SE3::from_eye_target_up(eye, target, up);
    let image_size : (usize, usize) = (1920, 1080);
    let K = Matrix33::new([[1200.0, 0.0, image_size.0 as f64/2.0],
                                            [0.0, 1200.0, image_size.1 as f64/2.0],
                                            [0.0, 0.0, 1.0]]);
    let camera = Camera::new(pose, image_size, K);

    //add elements
    let mut elements : Vec<Element> = Vec::new();
    let marble_image = image::open("examples/marble_texture.jpg").unwrap().to_rgb();
    elements.push(Element{
        name : format!("sphere1"),
        geometry : SceneGeometry::Sphere(Sphere::new(V3::new([2.0, 0.0, 1.5]), 1.5)),
        material : Material{coloration: Coloration::Texture(Texture{image : marble_image, tile : (4.0, 4.0)}),
             surface_type: SurfaceType::Diffuse,
             albedo: 0.5},
    });
    elements.push(Element{
        name : format!("sphere2"),
        geometry : SceneGeometry::Sphere(Sphere::new(V3::new([-2.0, 2.0, 1.0]), 1.0)),
        material : Material::color_with_defaults(Color::yellow()),
    });
    elements.push(Element{
        name : format!("sphere3"),
        geometry : SceneGeometry::Sphere(Sphere::new(V3::new([-2.5, -0.5, 0.5]), 0.5)),
        material : Material::color_with_defaults(Color::blue()),
    });

    let floor_image = image::open("examples/floor_texture.jpg").unwrap().to_rgb();
    elements.push(Element{
        name : format!("plane"),
        geometry : SceneGeometry::Plane(Plane::new(SE3::identity())),
        material : Material{coloration: Coloration::Texture(Texture{image : floor_image, tile : (2.0, 1.0)}),
             surface_type: SurfaceType::Diffuse,
             albedo: 0.5},
    });

    //add lights
    let mut lights : Vec<Light> = Vec::new();
    lights.push(Light::Directional(DirectionalLight{
        direction: V3::new([0.0, 0.0, -1.0]).normalize(),
        color: Color::yellow(),
        intensity: 0.3,
    }));
    lights.push(Light::Point(PointLight{
        position: V3::new([0.0, 0.0, 1.0]),
        color: Color::white(),
        intensity: 100.0,
    }));
    
    //build scene and render
    let scene = Scene{elements, lights};
    let depth_image = render_depth(&camera, &scene);
    let rgb_image = render_image(&camera, &scene);
    
    //save images to png
    depth_image.save("depth.png").unwrap();
    rgb_image.save("rgb.png").unwrap();
}