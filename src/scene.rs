use std::ops::{Add, Mul};
use image;
use image::{Pixel, Rgb, RgbImage};

use crate::EPSILON;
use crate::matrix::Matrix33;
use crate::vector::{V2,V3};
use crate::geometry::{Sphere, Plane, SO3, SE3, Ray};

const GAMMA: f32 = 2.2;

fn gamma_encode(linear: f32) -> f32 {
    linear.powf(1.0 / GAMMA)
}

fn gamma_decode(encoded: f32) -> f32 {
    encoded.powf(GAMMA)
}

fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub fn clamp(&self) -> Self {
        Color {
            r: clamp(self.r, 0.0, 1.0),
            g: clamp(self.g, 0.0, 1.0),
            b: clamp(self.b, 0.0, 1.0),
        } 
    }

    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Color {r: r, g: g, b: b}.clamp()
    }

    pub fn white() -> Self {
        Color::new(1.0, 1.0, 1.0)
    }

    pub fn yellow() -> Self {
        Color::new(1.0, 1.0, 0.0)
    }

    pub fn red() -> Self {
        Color::new(1.0, 0.0, 0.0)
    }

    pub fn green() -> Self {
        Color::new(0.0, 1.0, 0.0)
    }

    pub fn blue() -> Self {
        Color::new(0.0, 0.0, 1.0)
    }

    pub fn black() -> Self {
        Color::new(0.0, 0.0, 0.0)
    }

    pub fn random() -> Self {
        Color::new(rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>())
    }
}

impl Into<Rgb<u8>> for Color {
    fn into(self) -> Rgb<u8> {
        Rgb::from_channels(
            (gamma_encode(self.r) * 255.0) as u8,
            (gamma_encode(self.g) * 255.0) as u8,
            (gamma_encode(self.b) * 255.0) as u8,
            255,
        )
    }
}

impl From<Rgb<u8>> for Color {
    fn from(rgb: Rgb<u8>) -> Self {
        Color {
            r: gamma_decode(rgb.data[0] as f32 / 255.0),
            g: gamma_decode(rgb.data[1] as f32 / 255.0),
            b: gamma_decode(rgb.data[2] as f32 / 255.0),
        }
    }
}



impl Add for Color {
    type Output = Self;
    fn add(self, rhs: Color) -> Self::Output {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}
impl Mul for Color {
    type Output = Self;
    fn mul(self, rhs : Color) -> Self::Output {
        Color {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

impl Mul<Color> for f32 {
    type Output = Color;
    fn mul(self, rhs : Color) -> Self::Output {
        Color {
            r: self * rhs.r,
            g: self * rhs.g,
            b: self * rhs.b,
        }
    }
}

impl Mul<f32> for Color {
    type Output = Self;
    fn mul(self, rhs : f32) -> Self::Output {
        Color {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}

pub struct DirectionalLight {
    pub direction: V3,
    pub color: Color,
    pub intensity: f32,
}

pub struct PointLight {
    pub position: V3,
    pub color: Color,
    pub intensity: f32,
}

pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
}

#[allow(non_snake_case)]
pub struct Camera {
    /*
    camera axes are x right, y down, z forward
     */
    pub pose: SE3,
    pub inv_pose: SE3,
    pub image_size: (usize, usize), //width, height
    pub K: Matrix33,
    pub inv_K: Matrix33,
}

#[allow(non_snake_case)]
impl Camera {
    pub fn new(pose: SE3, image_size: (usize, usize), K: Matrix33) -> Self {
        Camera {
            pose: pose,
            inv_pose: pose.inverse(),
            image_size: image_size,
            K: K,
            inv_K: K.invert().unwrap(),
        }
    }

    pub fn project_point(&self, world_point : V3) -> V2 {
        let camera_point = self.inv_pose * world_point;
        let image_point = self.K * camera_point;
        V2::new([image_point[0] / image_point[2], image_point[1] / image_point[2]])
    }

    pub fn pixel_2_ray(&self, pixel : V2) -> Ray {
        let direction = self.inv_K * V3::new([pixel[0], pixel[1], 1.0]);
        let normalized_direction = direction.normalize();
        let mut ray = Ray::new(self.pose.t, normalized_direction);
        ray.direction = self.pose.r * ray.direction; //rotate ray direction into world frame
        ray
    }

    pub fn get_ray_bundle(&self) -> Vec<Vec<Ray>> {
        let mut rays = Vec::new();
        for x in 0..self.image_size.0 {
            let mut row_rays = Vec::new();
            for y in 0..self.image_size.1 {
                let pixel_center = V2::new([x as f64 + 0.5, y as f64 + 0.5]);
                row_rays.push(self.pixel_2_ray(pixel_center));
            }
            rays.push(row_rays);
        }
        rays
    }

}

#[derive(Debug)]
pub struct Texture {
    pub image : RgbImage,
    pub tile : (f32, f32),
}

#[derive(Debug)]
pub enum Coloration {
    Color(Color),
    Texture(Texture),
}

fn compute_rational_texture_coord(x: f32, width: f32) -> f32 {
    let wrapped_x = x % width;
    if wrapped_x < 0.0 {
        (wrapped_x + width)/width
    } else {
        wrapped_x/width
    }
}

fn bilinear_pixel_interpolation(xy : (f32, f32), image : &RgbImage) -> Rgb<u8> {
    let x: f32 = xy.0;
    let y = xy.1;

    let image_width = image.width();
    let image_height = image.height();

    //check if xy are integers
    if x.fract() == 0.0 && y.fract() == 0.0 {
        return *image.get_pixel(x as u32, y as u32);
    }
    else if x.fract() == 0.0 {
        let y1 = clamp(y.floor() as u32,0, image_height-1);
        let y2 = clamp(y.ceil() as u32, 0, image_height-1);
        let q1 = image.get_pixel(x as u32, y1);
        let q2 = image.get_pixel(x as u32, y2);
        let w1 = y2 as f32 - y;
        let w2 = y - y1 as f32;
        let r = w1 * q1.data[0] as f32 + w2 * q2.data[0] as f32;
        let g = w1 * q1.data[1] as f32 + w2 * q2.data[1] as f32;
        let b = w1 * q1.data[2] as f32 + w2 * q2.data[2] as f32;
        return Rgb([r.round() as u8, g.round() as u8, b.round() as u8]);
    } else if y.fract() == 0.0 {
        let x1 = clamp(x.floor() as u32,0,image_width-1);
        let x2 = clamp(x.ceil() as u32,0, image_width-1);
        let q1 = image.get_pixel(x1, y as u32);
        let q2 = image.get_pixel(x2, y as u32);
        let w1 = x2 as f32 - x;
        let w2 = x - x1 as f32;
        let r = w1 * q1.data[0] as f32 + w2 * q2.data[0] as f32;
        let g = w1 * q1.data[1] as f32 + w2 * q2.data[1] as f32;
        let b = w1 * q1.data[2] as f32 + w2 * q2.data[2] as f32;
        return Rgb([r.round() as u8, g.round() as u8, b.round() as u8]);
    }

    let x1 = clamp(x.floor() as u32,0, image_width-1);
    let y1 = clamp(y.floor() as u32,0, image_height-1);
    let x2 = clamp(x.ceil() as u32,0, image_width-1);
    let y2 = clamp(y.ceil() as u32,0, image_height-1);

    let q11 = image.get_pixel(x1, y1);
    let q12 = image.get_pixel(x1, y2);
    let q21 = image.get_pixel(x2, y1);
    let q22 = image.get_pixel(x2, y2);

    let w11 = (x2 as f32 - x) * (y2 as f32 - y);
    let w12 = (x2 as f32 - x) * (y - y1 as f32);
    let w21 = (x - x1 as f32) * (y2 as f32 - y);
    let w22 = (x - x1 as f32) * (y - y1 as f32);
    let denum = (x2 as f32 - x1 as f32) * (y2 as f32 - y1 as f32);

    let r = (w11 * q11.data[0] as f32 + w12 * q12.data[0] as f32 + w21 * q21.data[0] as f32 + w22 * q22.data[0] as f32) / denum;
    let g = (w11 * q11.data[1] as f32 + w12 * q12.data[1] as f32 + w21 * q21.data[1] as f32 + w22 * q22.data[1] as f32) / denum;
    let b = (w11 * q11.data[2] as f32 + w12 * q12.data[2] as f32 + w21 * q21.data[2] as f32 + w22 * q22.data[2] as f32) / denum;

    Rgb([r.round() as u8, g.round() as u8, b.round() as u8])
}

impl Coloration {
    pub fn color(&self, texture_coords: (f32, f32)) -> Color {
        match self {
            Coloration::Color(color) => *color,
            Coloration::Texture(texture) => {
                let r_x = compute_rational_texture_coord(texture_coords.0, texture.tile.0);
                let r_y = compute_rational_texture_coord(texture_coords.1, texture.tile.1);
                let x = r_x * (texture.image.width()-1) as f32;
                let y = r_y * (texture.image.height()-1) as f32;
                let pixel = bilinear_pixel_interpolation((x,y), &texture.image);
                // let pixel = texture.image.get_pixel(x as u32, y as u32);
                Color::from(pixel.to_rgb())
            }
        }
    }
}

#[derive(Debug)]
pub enum Material{
    Diffuse{albedo : f32, coloration : Coloration},
    Reflective{reflectivity : f32, albedo : f32, coloration : Coloration},
    Glass{index : f64, transparency : f32},
}

impl Material {
    pub fn defult_diffuse(color : Color) -> Self {
        Material::Diffuse { albedo: 0.18, coloration: Coloration::Color(color) }
    }
}

pub struct Intersection {
    pub time_of_flight : f64,
    pub normal : V3,
    pub point : V3,
    pub texture_coords : (f32, f32)
}

pub trait Intersectable {
    fn intersect(&self, ray : &Ray) -> Option<Intersection>;
    /*returns flight time, surface normals */
}

impl Intersectable for Plane{
    fn intersect(&self, ray : &Ray) -> Option<Intersection> {
        let denum = V3::dot(self.normal, ray.direction);
        if denum.abs() < EPSILON {
            return None;
        }
        let num = V3::dot(self.normal, self.center - ray.origin);
        let t = num / denum;
        if t > 0.0 {
            let intersection_point = ray.origin + t * ray.direction;
            let tex_x = (intersection_point[0] - self.center[0]) as f32;
            let tex_y = (intersection_point[1] - self.center[1]) as f32;
            return Some(Intersection{
                time_of_flight : t,
                normal : self.normal,
                point : intersection_point,
                texture_coords: (tex_x, tex_y),
            });
        }
        return None;
    }
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let l = self.center - ray.origin;
        
        let l_d_t = V3::dot(l, ray.direction);
        if l_d_t < 0.0 {
            return None;  // Sphere is behind the ray
        }

        let s2 = l.norm2() - l_d_t * l_d_t;
        let r2 = self.radius * self.radius;
        
        if s2 > r2 {
            return None;  // Ray misses the sphere
        }

        // Calculate two potential intersection times
        let dt = (r2 - s2).sqrt();
        let t0 = l_d_t - dt;
        let t1 = l_d_t + dt;

        // If the origin is inside the sphere, choose the larger t (exit point). 
        // Otherwise, choose the smaller t (entry point).
        let time_of_flight = if t0 < 0.0 { t1 } else { t0 };
        
        let normal = (ray.origin + time_of_flight * ray.direction - self.center).normalize();
        let intersection_point = ray.origin + time_of_flight * ray.direction;
        let phi = intersection_point[2].atan2(intersection_point[0]);
        let theta = (intersection_point[1]/self.radius).acos();
        let tex_x = ((1.0 + phi/std::f64::consts::PI) * 0.5) as f32; //normalize to [0,1]
        let tex_y = (theta/std::f64::consts::PI) as f32; // normalize to [0,1]

        Some(Intersection{
            time_of_flight: time_of_flight,
            normal: normal,
            point: intersection_point,
            texture_coords: (tex_x, tex_y),
        })
    }
}

#[derive(Debug)]
pub enum SceneGeometry{
    Sphere(Sphere),
    Plane(Plane)
}

impl Intersectable for SceneGeometry{
    fn intersect(&self, ray : &Ray) -> Option<Intersection> {
        match self {
            SceneGeometry::Sphere(sphere) => sphere.intersect(ray),
            SceneGeometry::Plane(plane) => plane.intersect(ray)
        }
    }
}

#[derive(Debug)]
pub struct Element {
    pub name : String,
    pub geometry: SceneGeometry,
    pub material: Material
}

pub struct Scene {
    pub elements: Vec<Element>,
    pub lights : Vec<Light>,
}

impl Scene {
    pub fn cast(&self, ray: &Ray) -> Option<Intersection> {
        let mut closest_intersection: Option<Intersection> = None;
    
        for element in &self.elements {
            if let Some(current_intersection) = element.geometry.intersect(ray) {
                match &closest_intersection {
                    Some(closest) => {
                        if current_intersection.time_of_flight < closest.time_of_flight {
                            closest_intersection = Some(current_intersection);
                        }
                    },
                    None => {
                        closest_intersection = Some(current_intersection);
                    }
                }
            }
        }
        closest_intersection
    }
}

#[cfg(test)]
pub mod tests{
    use super::*;

    #[test]
    fn test_color() {
        let red = Color::new(1.0, 0.0, 0.0);
        let blue = Color::new(0.0, 0.0, 1.0);
        let purple = Color::new(1.0, 0.0, 1.0);
        let black = Color::default();
        let half_green = Color::new(0.0, 0.5, 0.0);
        let green = Color::new(0.0, 1.0, 0.0);
        
        //black
        assert_eq!(black, Color::new(0.0, 0.0, 0.0));
        assert_eq!(red * black, black);

        //addition
        assert_eq!(red + blue, purple);
        assert_eq!(blue + red, red + blue);

        //multiplication
        assert_eq!(red * blue, black);
        assert_eq!(half_green * 2.0, green);

        //io into RGB
        let red_rgb: Rgb<u8> = red.into();
        assert_eq!(red_rgb, Rgb::from_channels(255, 0, 0, 255));
        let red_color: Color = red_rgb.into();
        assert_eq!(red_color, red);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_camera() {
        let pose = SE3::new(SO3::identity(), V3::new([1.0, 1.0, 0.0]));
        let image_size = (640, 480);
        let K = Matrix33::new([[1000.0, 0.0, 320.0],
                               [0.0, 1000.0, 240.0],
                               [0.0, 0.0, 1.0]]);
        let camera = Camera::new(pose, image_size, K);
        
        //point right infront of camera
        let point = V3::new([1.0, 1.0, 10.0]);
        let image_point = camera.project_point(point);
        assert!(image_point[0] == K[0][2]);
        assert!(image_point[1] == K[1][2]);
        
        let point = V3::new([5.0, 5.0, 10.0]);
        let image_point = camera.project_point(point);
        assert!(image_point[0] > K[0][2]);
        assert!(image_point[1] > K[1][2]);

        let expected_direction = (point - camera.pose.t).normalize();
        let ray = camera.pixel_2_ray(image_point);
        assert!(V3::is_close(&ray.direction, &expected_direction, None));
    }

    #[test]
    fn test_inside_sphere_intersection() {
        let sphere = Sphere::new(V3::default(),2.0);
        let eye = V3::new([0.0,0.0,1.0]);
        let target = V3::new([2.0,0.0,0.0]);
        let ray = Ray::new(eye, (target-eye).normalize());
        let intersection = sphere.intersect(&ray).unwrap();
        assert!(V3::is_close(&intersection.point, &V3::new([2.0,0.0,0.0]), None));
    }

    #[test]
    fn test_outside_sphere_intersection() {
        let sphere = Sphere::new(V3::zeros(),2.0);
        let eye = V3::new([0.0,0.0,3.0]);
        let target = V3::new([0.0,0.0,0.0]);
        let ray = Ray::new(eye, (target-eye).normalize());
        let intersection = sphere.intersect(&ray).unwrap();
        assert!(V3::is_close(&intersection.point, &V3::new([0.0,0.0,2.0]), None));
    }
}
