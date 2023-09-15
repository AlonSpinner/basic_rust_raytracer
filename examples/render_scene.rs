use raytracing_tutorial::geometry::{Sphere, SE3, Plane};
use raytracing_tutorial::matrix::Matrix33;
use raytracing_tutorial::vector::{V3};
use raytracing_tutorial::scene::{Camera, Scene, Element, Material, SceneGeometry, SurfaceType, Color};
use raytracing_tutorial::rendering::{render_depth,render_image};
use rand;

#[allow(non_snake_case)]
fn main() {
    //create camera
    let eye = V3::new([0.0, 0.0, 5.0]);
    let target = V3::new([0.0, 0.0, 0.0]);
    let up = V3::new([0.0, 1.0, 0.0]);
    let pose = SE3::from_eye_target_up(eye, target, up);
    let image_size : (usize, usize) = (1920/2, 1080/2);
    let K = Matrix33::new([[800.0, 0.0, image_size.0 as f64/2.0],
                          [0.0, 800.0, image_size.1 as f64/2.0],
                          [0.0, 0.0, 1.0]]);
    let camera = Camera::new(pose, image_size, K);


    let radius = [0.3,0.5];
    let box_x = [-2.0, 2.0];
    let box_y = [-2.0, 2.0];
    let box_z = [0.0, 2.0];

    let mut elements : Vec<Element> = Vec::new();
    for i in 0..20 {
        let x = box_x[0] + (box_x[1] - box_x[0]) * rand::random::<f64>();
        let y = box_y[0] + (box_y[1] - box_y[0]) * rand::random::<f64>();
        let z = box_z[0] + (box_z[1] - box_z[0]) * rand::random::<f64>();
        let r = radius[0] + (radius[1] - radius[0]) * rand::random::<f64>();
        let sphere = Element{
            name : format!("sphere{}", i),
            geometry : SceneGeometry::Sphere(Sphere::new(V3::new([x, y, z]), r)),
            material : Material { color: Color::red(), albedo: 0.0, surface_type: SurfaceType::Diffuse },
        };
        elements.push(sphere);
    }

    // add xy plane
    let plane = Element{
        name : format!("plane"),
        geometry : SceneGeometry::Plane(Plane::new(V3::default(), V3::new([0.0, 0.0, 1.0]))),
        material : Material { color: Color::green(), albedo: 0.0, surface_type: SurfaceType::Diffuse },
    };
    elements.push(plane);
    
    let scene = Scene::new(elements);
    let depth_image = render_depth(&camera, &scene);
    let rgb_image = render_image(&camera, &scene);
    
    //save images to png
    depth_image.save("depth.png").unwrap();
    rgb_image.save("rgb.png").unwrap();
}