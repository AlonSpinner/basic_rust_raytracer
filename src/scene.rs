use std::ops::{Add, Mul};
use image;
use image::{Pixel, Rgba};

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

fn clamp(x: f32, min: f32, max: f32) -> f32 {
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
    fn new(r: f32, g: f32, b: f32) -> Self {
        Color {r: clamp(r,0.0,1.0),
               g: clamp(g,0.0,1.0),
               b: clamp(b,0.0,1.0) }
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

    pub fn random() -> Self {
        Color::new(rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>())
    }
}

impl Into<Rgba<u8>> for Color {
    fn into(self) -> Rgba<u8> {
        Rgba::from_channels(
            (gamma_encode(self.r) * 255.0) as u8,
            (gamma_encode(self.g) * 255.0) as u8,
            (gamma_encode(self.b) * 255.0) as u8,
            255,
        )
    }
}

impl From<Rgba<u8>> for Color {
    fn from(rgba: Rgba<u8>) -> Self {
        Color {
            r: gamma_decode(rgba.data[0] as f32 / 255.0),
            g: gamma_decode(rgba.data[1] as f32 / 255.0),
            b: gamma_decode(rgba.data[2] as f32 / 255.0),
        }
    }
}

impl Add for Color {
    type Output = Self;
    fn add(self, rhs: Color) -> Self::Output {
        Color {
            r: clamp(self.r + rhs.r,0.0,1.0),
            g: clamp(self.g + rhs.g,0.0,1.0),
            b: clamp(self.b + rhs.b,0.0,1.0),
        }
    }
}
impl Mul for Color {
    type Output = Self;
    fn mul(self, rhs : Color) -> Self::Output {
        Color {
            r: clamp(self.r * rhs.r,0.0,1.0),
            g: clamp(self.g * rhs.g,0.0,1.0),
            b: clamp(self.b * rhs.b,0.0,1.0),
        }
    }
}

impl Mul<Color> for f32 {
    type Output = Color;
    fn mul(self, rhs : Color) -> Self::Output {
        Color {
            r: clamp(self * rhs.r,0.0,1.0),
            g: clamp(self * rhs.g,0.0,1.0),
            b: clamp(self * rhs.b,0.0,1.0),
        }
    }
}

impl Mul<f32> for Color {
    type Output = Self;
    fn mul(self, rhs : f32) -> Self::Output {
        Color {
            r: clamp(self.r * rhs,0.0,1.0),
            g: clamp(self.g * rhs,0.0,1.0),
            b: clamp(self.b * rhs,0.0,1.0),
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
pub enum SurfaceType {
    Diffuse,
    Reflective,
    Refractive
}

#[derive(Debug)]
pub struct Material{
    pub color : Color,
    pub albedo : f32,
    pub surface_type : SurfaceType
}

impl Material {
    pub fn color_with_defaults(color : Color) -> Self {
        Material {
            color : color,
            albedo : 0.18,
            surface_type : SurfaceType::Diffuse
        }
    }
}

pub struct Intersection {
    pub time_of_flight : f64,
    pub normal : V3,
    pub point : V3,
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
            return Some(Intersection{
                time_of_flight : t,
                normal : self.normal,
                point : ray.origin + t * ray.direction
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

        let time_of_flight = l_d_t - (r2-s2).sqrt();
        let normal = (ray.origin + time_of_flight * ray.direction - self.center).normalize();

        Some(Intersection{
            time_of_flight: time_of_flight,
            normal: normal,
            point: ray.origin + time_of_flight * ray.direction,
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
    pub fn new(elements : Vec<Element>, lights : Vec<Light>) -> Self {
        Scene {
            //lights one day
            elements: elements,
            lights : lights,
        }
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
        assert_eq!(red + blue + blue, red + blue);

        //multiplication
        assert_eq!(red * blue, black);
        assert_eq!(red * 2.0, red);
        assert_eq!(half_green * 2.0, green);

        //io into RGBA
        let red_rgba: Rgba<u8> = red.into();
        assert_eq!(red_rgba, Rgba::from_channels(255, 0, 0, 255));
        let red_color: Color = red_rgba.into();
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
}
